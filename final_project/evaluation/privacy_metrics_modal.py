import modal
import boto3
import os
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# -----------------------------
# Modal GPU image
# -----------------------------
image = (
    modal.Image.debian_slim()
    .pip_install("boto3", "numpy", "scikit-learn", "torch")
)

app = modal.App("ehr-privacy-gpu")

def log(msg):
    print(f"[LOG] {msg}")

# -----------------------------
# Helper: download .npy from S3
# -----------------------------
def download_s3_file(s3, bucket, key, local_path):
    log(f"Downloading s3://{bucket}/{key}")
    s3.download_file(bucket, key, local_path)
    return np.load(local_path)

# -----------------------------
# ATTRIBUTE INFERENCE (CPU)
# -----------------------------
def attribute_inference_f1(real_train, synthetic, known_idx, unknown_idx, k=1):
    log("Starting Attribute Inference Attack (CPU)...")
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(synthetic[:, known_idx])

    _, indices = knn.kneighbors(real_train[:, known_idx])

    preds = []
    for i in range(real_train.shape[0]):
        if i % 500 == 0:
            log(f"  Processing record {i}/{real_train.shape[0]}")
        neigh = indices[i]
        vote = synthetic[neigh][:, unknown_idx].mean(axis=0) > 0.5
        preds.append(vote.astype(int))

    preds = np.array(preds)
    true_vals = real_train[:, unknown_idx]
    f1 = f1_score(true_vals.flatten(), preds.flatten(), zero_division=0)
    log(f"Attribute Inference F1: {f1:.4f}")
    return f1

# -----------------------------
# MEMBERSHIP INFERENCE (GPU)
# -----------------------------
def membership_inference_f1(real_train, real_test, synthetic, threshold):
    log("Starting Membership Inference Attack (GPU)...")

    def gpu_min_distances(real, synthetic):
        real_t = torch.tensor(real, dtype=torch.float32, device="cuda")
        syn_t  = torch.tensor(synthetic, dtype=torch.float32, device="cuda")
        dists = torch.norm(real_t[:, None, :] - syn_t[None, :, :], dim=2)
        mins = dists.min(dim=1).values
        return mins.cpu().numpy()

    log("Computing distances for train samples...")
    train_dists = gpu_min_distances(real_train, synthetic)
    log("Computing distances for test samples...")
    test_dists  = gpu_min_distances(real_test, synthetic)

    train_pred = (train_dists < threshold).astype(int)
    test_pred  = (test_dists  < threshold).astype(int)

    preds = np.concatenate([train_pred, test_pred])
    true  = np.concatenate([np.ones(len(train_pred)), np.zeros(len(test_pred))])

    f1 = f1_score(true, preds)
    log(f"Membership Inference F1: {f1:.4f}")
    return f1

# -----------------------------
# MAIN MODAL FUNCTION
# -----------------------------
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("aws-secret")],
    gpu="H100",
    timeout=60 * 60 * 2,   # 2 hours
)
def run():
    BUCKET = "healthforge-final-bucket"
    THRESHOLD = 100.0
    SAMPLE_SIZE = 10000

    # Setup S3
    s3 = boto3.client("s3")

    # -----------------------------
    # Download the real and synthetic files
    # -----------------------------
    real_file = "original_vectors.npy"
    synthetic_file = "sample_output.npy"

    full_real_data = download_s3_file(s3, BUCKET, real_file, f"/tmp/{real_file}")
    synthetic = download_s3_file(s3, BUCKET, synthetic_file, f"/tmp/{synthetic_file}")

    log(f"Full real data shape: {full_real_data.shape}")
    log(f"Synthetic data shape: {synthetic.shape}")

    # Randomly sample 10,000 rows from real data
    if full_real_data.shape[0] > SAMPLE_SIZE:
        indices = np.random.choice(full_real_data.shape[0], SAMPLE_SIZE, replace=False)
        full_real_data = full_real_data[indices]
    log(f"Sampled real data shape: {full_real_data.shape}")

    # Train/test split on real data
    real_train, real_test = train_test_split(full_real_data, test_size=0.2, random_state=42)
    log(f"Train shape: {real_train.shape}, Test shape: {real_test.shape}")

    # Define known/unknown features for attribute inference
    known_idx = np.arange(256)
    unknown_idx = np.arange(256, min(600, full_real_data.shape[1]))

    # Attribute inference
    attr_f1 = attribute_inference_f1(real_train, synthetic, known_idx, unknown_idx)

    # Membership inference
    memb_f1 = membership_inference_f1(real_train, real_test, synthetic, THRESHOLD)

    log("---------- FINAL RESULTS ----------")
    log(f"Attribute Inference F1:   {attr_f1}")
    log(f"Membership Inference F1: {memb_f1}")

    return {
        "attribute_inference_f1": float(attr_f1),
        "membership_inference_f1": float(memb_f1)
    }

# -----------------------------
# Auto-run when script executed
# -----------------------------
@app.local_entrypoint()
def main():
    run.remote()
    print("Submitted privacy metrics job to Modal…")

if __name__ == "__main__":
    with app.run():
        main()
