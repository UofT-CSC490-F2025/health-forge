import modal
import boto3
import tempfile
import os
import random
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
BUCKET = "healthforge-final-bucket-1"
DATA_KEY = "original_vectors_gemma.npy"
EMBEDS_KEY = "vector_tag_embeddings_gemma.npy"

MODEL_OUTPUT_KEY = "results/best_diffusion_model_WITH_ENCODER_TEST.pt"
AUTOENCODER_KEY = "autoencoder/best_autoencoder_model.pt"
LATENT_MEAN_KEY = "results/latent_mean_TEST.npy"
LATENT_STD_KEY = "results/latent_std_TEST.npy"
RESUME = False

app = modal.App("diffusion-training-app")

image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch",
        "numpy",
        "boto3",
        "pyyaml",
        "sentence-transformers",
        "tqdm",
        "transformers",
        "scipy",
        "scikit-learn",
        "pandas"
    ])
    .add_local_file("train.py", "/root/train.py")
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("trainer.py", "/root/trainer.py")
    .add_local_file("data_utils.py", "/root/data_utils.py")
    .add_local_file("configs.yaml", "/root/configs.yaml")
    .add_local_file("autoencoder.py", "/root/autoencoder.py")
)

aws_secret = modal.Secret.from_name("aws-secret")

# ---------------------------
# GPU TRAINING WORKER
# ---------------------------
@app.function(
    gpu=["H100"],       # single-GPU container
    timeout=20*60*60,
    image=image,
    secrets=[aws_secret]
)
def train_worker():
    """
    Load training data and configuration, run model training, and upload outputs to S3.

    Downloads samples, text embeddings, and an autoencoder checkpoint from S3; loads
    a training configuration; optionally resumes from a previous model checkpoint; and
    invokes `train_from_pkl` to train a diffusion model. After training, the function
    uploads the best model and latent statistics back to S3 and cleans up temporary files.

    Returns:
        None. All results are saved to S3.
    """
    import torch
    import yaml
    from train import train_from_pkl

    print("GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load configs
    cfg_path = "/root/configs.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("Downloading samples and text embeddings from S3...")
    s3 = boto3.client("s3")
    data_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    s3.download_fileobj(BUCKET, DATA_KEY, data_tmp)
    data_tmp.close()

    text_embeds_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    s3.download_fileobj(BUCKET, EMBEDS_KEY, text_embeds_tmp)
    text_embeds_tmp.close()

    print("Loading samples...")
    samples = np.load(data_tmp.name)
    print("Loading text_embeds...")
    text_embeds = np.load(text_embeds_tmp.name)

    print("Loading autoencoder from s3")
    autoencoder_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    s3.download_fileobj(BUCKET, AUTOENCODER_KEY, autoencoder_tmp)
    autoencoder_tmp.close()
    ae_path = autoencoder_tmp.name


    print("Loaded dataset:")
    print("samples:", samples.shape)
    print("text_embeds:", text_embeds.shape)
    print(text_embeds[0])


    resume_ckpt = None
    if RESUME:
        ckpt_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        s3.download_fileobj(BUCKET, MODEL_OUTPUT_KEY, ckpt_tmp)
        ckpt_tmp.close()
        resume_ckpt = ckpt_tmp.name

    # Train
    save_path, latent_mean_path, latent_std_path = train_from_pkl(
        cfg,
        samples,
        text_embeds,
        autoencoder_path=ae_path,
        save_path="/root/best_diffusion_model.pt",
        latent_mean_path='/root/latent_mean.npy',
        latent_std_path = '/root/latent_std.npy',
        resume_ckpt=resume_ckpt
    )

    if os.path.exists(save_path):
        print("Uploading best model to S3...")
        s3.upload_file(save_path, BUCKET, MODEL_OUTPUT_KEY)
        s3.upload_file(latent_mean_path, BUCKET, LATENT_MEAN_KEY)
        s3.upload_file(latent_std_path, BUCKET, LATENT_STD_KEY)
        print("Upload complete.")
    else:
        print("ERROR: best model file not found")

    os.unlink(data_tmp.name)
    os.unlink(text_embeds_tmp.name)



# ---------------------------
# ENTRYPOINT
# ---------------------------
@app.local_entrypoint()
def main():
    h = train_worker.spawn()
    h.get()
    print("Training complete.")

if __name__ == "__main__":
    with app.run():
        main()
