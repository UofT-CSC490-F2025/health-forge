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
    timeout=60*60
)
def split_real_data_s3():
    # -----------------------------
    # Configuration
    # -----------------------------
    BUCKET = "healthforge-final-bucket"

    # Main patient embeddings
    REAL_FILE_KEY = "original_vectors_gemma.npy"
    TRAIN_FILE_KEY = "original_vectors_train.npy"
    VAL_FILE_KEY = "original_vectors_val.npy"

    # Second file (tag embeddings for each patient)
    TAG_EMBED_KEY = "vector_tag_embeddings_gemma.npy"
    TAG_EMBED_TRAIN_KEY = "vector_tag_embeddings_train.npy"
    TAG_EMBED_VAL_KEY = "vector_tag_embeddings_val.npy"

    # Third file (tag IDs for each patient)
    TAG_IDS_KEY = "vector_tags_gemma.npy"
    TAG_IDS_TRAIN_KEY = "vector_tags_train.npy"
    TAG_IDS_VAL_KEY = "vector_tags_val.npy"

    LOCAL_DIR = "/tmp"

    # -----------------------------
    # Setup S3
    # -----------------------------
    s3 = boto3.client("s3")

    # Helper to download and load
    def load_from_s3(key):
        local_path = os.path.join(LOCAL_DIR, os.path.basename(key))
        log(f"Downloading s3://{BUCKET}/{key} → {local_path}")
        s3.download_file(BUCKET, key, local_path)
        return np.load(local_path)

    # -----------------------------
    # Download all three datasets
    # -----------------------------
    full_data = load_from_s3(REAL_FILE_KEY)              # Patients
    full_tag_embeds = load_from_s3(TAG_EMBED_KEY)        # Tag embeddings
    full_tag_ids = load_from_s3(TAG_IDS_KEY)             # Tag ID sequences

    log(f"original_vectors_gemma shape: {full_data.shape}")
    log(f"vector_tag_embeddings_gemma shape: {full_tag_embeds.shape}")
    log(f"vector_tags_gemma shape: {full_tag_ids.shape}")

    # -----------------------------
    # Sanity check: row alignment
    # -----------------------------
    N = full_data.shape[0]
    if full_tag_embeds.shape[0] != N or full_tag_ids.shape[0] != N:
        raise ValueError("Row count mismatch between the datasets — alignment lost.")

    # -----------------------------
    # Train/Validation split
    # -----------------------------
    indices = np.arange(N)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Split all three datasets identically
    train_data     = full_data[train_idx]
    val_data       = full_data[val_idx]

    train_tag_emb  = full_tag_embeds[train_idx]
    val_tag_emb    = full_tag_embeds[val_idx]

    train_tag_ids  = full_tag_ids[train_idx]
    val_tag_ids    = full_tag_ids[val_idx]

    log(f"Train shape: {train_data.shape}, Validation shape: {val_data.shape}")
    log(f"Train TAG EMB shape: {train_tag_emb.shape}, Validation TAG EMB shape: {val_tag_emb.shape}")
    log(f"Train TAG IDS shape: {train_tag_ids.shape}, Validation TAG IDS shape: {val_tag_ids.shape}")

    # -----------------------------
    # Save all splits
    # -----------------------------
    def save_local(fname, arr):
        path = os.path.join(LOCAL_DIR, fname)
        np.save(path, arr)
        return path

    train_path     = save_local("train_split.npy", train_data)
    val_path       = save_local("val_split.npy", val_data)

    tag_emb_train_path = save_local("tag_emb_train_split.npy", train_tag_emb)
    tag_emb_val_path   = save_local("tag_emb_val_split.npy", val_tag_emb)

    tag_ids_train_path = save_local("tag_ids_train_split.npy", train_tag_ids)
    tag_ids_val_path   = save_local("tag_ids_val_split.npy", val_tag_ids)

    # -----------------------------
    # Upload to S3
    # -----------------------------
    uploads = [
        (train_path, TRAIN_FILE_KEY),
        (val_path, VAL_FILE_KEY),
        (tag_emb_train_path, TAG_EMBED_TRAIN_KEY),
        (tag_emb_val_path, TAG_EMBED_VAL_KEY),
        (tag_ids_train_path, TAG_IDS_TRAIN_KEY),
        (tag_ids_val_path, TAG_IDS_VAL_KEY),
    ]

    for local_path, s3_key in uploads:
        log(f"Uploading {local_path} → s3://{BUCKET}/{s3_key}")
        s3.upload_file(local_path, BUCKET, s3_key)

    log("Done! All 3 datasets split and uploaded with aligned indices.")

    return {
        "train": train_data.shape,
        "val": val_data.shape,
        "tag_embed_train": train_tag_emb.shape,
        "tag_embed_val": val_tag_emb.shape,
        "tag_ids_train": train_tag_ids.shape,
        "tag_ids_val": val_tag_ids.shape,
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
