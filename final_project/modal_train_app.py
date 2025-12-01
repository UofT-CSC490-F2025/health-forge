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
MERGED_KEY = "data/final_merged.pkl"
DATA_KEY = "original_vectors_gemma.npy"
EMBEDS_KEY = "vector_tag_embeddings_gemma.npy"

MODEL_OUTPUT_KEY = "results/best_diffusion_model_WITH_ENCODER.pt"

AUTOENCODER_KEY = "autoencoder/best_autoencoder_model.pt"

LATENT_MEAN_KEY = "results/latent_mean.npy"

LATENT_STD_KEY = "results/latent_std.npy"

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
    import torch
    import yaml
    from train import train_from_pkl
    from autoencoder import EHRLatentAutoencoder

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

    INPUT_DIM = 1806
    LATENT_DIM = 1024

    autoencoder = EHRLatentAutoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to("cuda")
    state_dict = torch.load(ae_path, map_location="cuda")
   
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()

     # Freeze AE weights
    for p in autoencoder.parameters():
        p.requires_grad = False

    print("Autoencoder loaded.")

    #Normalize input vector
    oldest_age = samples[:, 1].max()
    max_admissions = samples[:, 3].max()

    samples[:, 1] = samples[:, 1] / oldest_age
    samples[:, 3] = samples[:, 3] / max_admissions
    assert (samples.max() <= 1.0) and (samples.min() >= 0), "Samples are not in a normalized range"


    #Normalize text embedding
    norms = text_embeds.norm(dim=1, keepdim=True) + 1e-8
    text_embeds = text_embeds / norms

    B = 2048  # batch size
    latents = []

    samples_t = torch.from_numpy(samples).float().to("cuda")  # put entire dataset on GPU

    with torch.no_grad():
        for i in range(0, samples_t.size(0), B):
            batch = samples_t[i:i+B]          # (B, 1806)

            # forward AE: returns (logits, latent)
            _, z = autoencoder(batch)         # z is (B, 1024)

            latents.append(z.cpu())           # store on CPU to avoid GPU blow-up

    latents = torch.cat(latents, dim=0)       # final shape: (N, 1024)
    latents = latents.detach().cpu().numpy()

    latent_mean = latents.mean(0)
    latent_std  = latents.std(0) + 1e-6
    latents = (latents - latent_mean) / latent_std

    np.save("latent_mean.npy", latent_mean)
    np.save("latent_std.npy", latent_std)

    print("Loaded dataset:")
    print("samples:", samples.shape)
    print("latents:", latents.shape )
    print(" text_embeds:", text_embeds.shape)

    
    
    resume_ckpt = None
    if RESUME:
        ckpt_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        s3.download_fileobj(BUCKET, MODEL_OUTPUT_KEY, ckpt_tmp)
        ckpt_tmp.close()
        resume_ckpt = ckpt_tmp.name

    # Train
    train_from_pkl(
        cfg,
        # pkl_path=pkl_tmp.name,
        latents,
        text_embeds,
        save_path="/root/best_diffusion_model.pt",
        resume_ckpt=resume_ckpt
    )

    # Upload the trained model
    best_model_path = "/root/best_diffusion_model.pt"
    latent_mean_path = '/root/latent_mean.npy'
    latent_std_path = '/root/latent_std.npy'
    if os.path.exists(best_model_path):
        print("Uploading best model to S3...")
        s3.upload_file(best_model_path, BUCKET, MODEL_OUTPUT_KEY)
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
