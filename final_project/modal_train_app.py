import modal
import boto3
import tempfile
import os
import random
import socket
import time

# ---------------------------
# CONFIG
# ---------------------------
BUCKET = "healthforge-final-bucket"
VECTORS_KEY = "original_vectors.npy"
EMBEDS_KEY = "vector_tag_embeddings.npy"
MODEL_OUTPUT_KEY = "results/best_diffusion_model.pt"

NUM_WORKERS = 8  # number of containers / GPUs

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
    gpu=["H100"],  # Request 8 GPUs in **one container**
    timeout=20*60*60,
    image=image,
    secrets=[aws_secret]
)
def train_worker(rank: int = 0, world_size: int = 8):
    import torch
    import yaml
    import tempfile
    import os
    import numpy as np
    from train import train_from_arrays

    # GPU check
    print("GPUs available in container:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load configs
    cfg_path = "/root/configs.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Download dataset from S3
    s3 = boto3.client("s3")

    vec_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    s3.download_fileobj(BUCKET, VECTORS_KEY, vec_tmp)
    vec_tmp.close()
    vectors = np.load(vec_tmp.name)

    emb_tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    s3.download_fileobj(BUCKET, EMBEDS_KEY, emb_tmp)
    emb_tmp.close()
    text_embeds = np.load(emb_tmp.name)

    N = 100000   # choose your reduced dataset size
    vectors = vectors[:N]
    text_embeds = text_embeds[:N]
    print("Dataset reduced to:", len(vectors))

    # Train with DDP or single-machine multi-GPU
    train_from_arrays(
        cfg, vectors, text_embeds,
        save_path="/root/best_diffusion_model.pt"  # ensure correct filename
    )

    # Upload best model (rank 0 only)
    best_model_path = "/root/best_diffusion_model.pt"

    if rank == 0:
        if os.path.exists(best_model_path):
            print("Uploading best model to S3...")
            s3.upload_file(best_model_path, BUCKET, MODEL_OUTPUT_KEY)
            print("Upload complete.")
        else:
            print("ERROR: best model file not found:", best_model_path)

    # Cleanup
    os.unlink(vec_tmp.name)
    os.unlink(emb_tmp.name)


# ---------------------------
# ORCHESTRATOR FUNCTION
# ---------------------------
@app.local_entrypoint()
def main():
    # run training in the modal container
    h = train_worker.spawn(rank=0, world_size=8)
    h.get()
    print("Distributed training complete.")

if __name__ == "__main__":
    with app.run():
        main()
