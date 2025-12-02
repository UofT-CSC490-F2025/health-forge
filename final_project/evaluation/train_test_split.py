import modal
import boto3
import numpy as np
from sklearn.model_selection import train_test_split
import os

# -----------------------------
# Modal image
# -----------------------------
image = (
    modal.Image.debian_slim()
    .pip_install("boto3", "numpy", "scikit-learn")
)

app = modal.App("ehr-data-split")

def log(msg):
    print(f"[LOG] {msg}")

# -----------------------------
# Modal function
# -----------------------------
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=60*60  # 1 hour should be enough
)
def split_real_data_s3():
    # -----------------------------
    # Configuration
    # -----------------------------
    BUCKET = "healthforge-final-bucket"
    REAL_FILE_KEY = "original_vectors_gemma.npy"
    TRAIN_FILE_KEY = "original_vectors_train.npy"
    VAL_FILE_KEY = "original_vectors_val.npy"
    LOCAL_DIR = "/tmp"

    # -----------------------------
    # Setup S3
    # -----------------------------
    s3 = boto3.client("s3")

    # -----------------------------
    # Download full dataset
    # -----------------------------
    local_real_path = os.path.join(LOCAL_DIR, os.path.basename(REAL_FILE_KEY))
    log(f"Downloading s3://{BUCKET}/{REAL_FILE_KEY} to {local_real_path}...")
    s3.download_file(BUCKET, REAL_FILE_KEY, local_real_path)

    full_data = np.load(local_real_path)
    log(f"Full dataset shape: {full_data.shape}")

    # -----------------------------
    # Train/Validation split
    # -----------------------------
    train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=42)
    log(f"Train shape: {train_data.shape}, Validation shape: {val_data.shape}")

    # -----------------------------
    # Save locally
    # -----------------------------
    train_path = os.path.join(LOCAL_DIR, "train_split.npy")
    val_path = os.path.join(LOCAL_DIR, "val_split.npy")
    np.save(train_path, train_data)
    np.save(val_path, val_data)

    # -----------------------------
    # Upload to S3
    # -----------------------------
    log(f"Uploading train split to s3://{BUCKET}/{TRAIN_FILE_KEY}...")
    s3.upload_file(train_path, BUCKET, TRAIN_FILE_KEY)

    log(f"Uploading validation split to s3://{BUCKET}/{VAL_FILE_KEY}...")
    s3.upload_file(val_path, BUCKET, VAL_FILE_KEY)

    log("Done! Train and validation splits uploaded to S3.")
    return {
        "train_shape": train_data.shape,
        "val_shape": val_data.shape
    }

# -----------------------------
# Local entrypoint
# -----------------------------
@app.local_entrypoint()
def main():
    split_real_data_s3.remote()
    print("Submitted dataset splitting job to Modal…")

if __name__ == "__main__":
    with app.run():
        main()
