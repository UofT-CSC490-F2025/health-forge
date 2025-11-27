import modal
import boto3
import tempfile
import os
import random

# ---------------------------
# CONFIG
# ---------------------------
BUCKET = "healthforge-final-bucket"
MERGED_KEY = "data/final_merged.pkl"
MODEL_OUTPUT_KEY = "results/best_diffusion_model.pt"

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
    import pickle
    from train import train_from_pkl

    print("GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load configs
    cfg_path = "/root/configs.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Download PKL
    print("Downloading final_merged.pkl from S3...")
    s3 = boto3.client("s3")
    pkl_tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    s3.download_fileobj(BUCKET, MERGED_KEY, pkl_tmp)
    pkl_tmp.close()

    print("Loading PKL...")
    with open(pkl_tmp.name, "rb") as f:
        merged_data = pickle.load(f)

    samples = merged_data["samples"]
    text_embeds = merged_data["text_embeds"]

    print("Loaded dataset:")
    print(" samples:", samples.shape)
    print(" text_embeds:", text_embeds.shape)

    # Apply truncation
    num_samples = cfg.get("num_samples", None)
    if num_samples:
        samples = samples[:num_samples]
        text_embeds = text_embeds[:num_samples]
        print(f"Dataset truncated to {num_samples} samples")

        # MUST overwrite PKL for train_from_pkl to use truncated data
        new_data = {
            "samples": samples,
            "text_embeds": text_embeds
        }
        with open(pkl_tmp.name, "wb") as f:
            pickle.dump(new_data, f)

    # Train
    train_from_pkl(
        cfg,
        pkl_path=pkl_tmp.name,
        save_path="/root/best_diffusion_model.pt"
    )

    # Upload the trained model
    best_model_path = "/root/best_diffusion_model.pt"
    if os.path.exists(best_model_path):
        print("Uploading best model to S3...")
        s3.upload_file(best_model_path, BUCKET, MODEL_OUTPUT_KEY)
        print("Upload complete.")
    else:
        print("ERROR: best model file not found")

    # Cleanup
    os.unlink(pkl_tmp.name)


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
