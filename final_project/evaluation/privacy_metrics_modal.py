import modal
import boto3
import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# -----------------------------
# Modal GPU image
# -----------------------------
image = (
    modal.Image.debian_slim()
    .pip_install("boto3", "numpy", "scikit-learn", "torch", "pandas")
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
# Helper: download column labels from S3
# -----------------------------
def download_column_labels(s3, bucket, key):
    log(f"Downloading column labels from s3://{bucket}/{key}")
    local_path = f"/tmp/{os.path.basename(key)}"
    s3.download_file(bucket, key, local_path)
    df = pd.read_csv(local_path, header=None)
    return df.iloc[0].tolist()  # single row of column names

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
    f1 = f1_score(true_vals.flatten(), preds.flatten(), zero_division=0, average="micro")
    log(f"Attribute Inference F1: {f1:.4f}")

    # -------------------------
    # DEBUG: Attribute inference sanity checks
    # -------------------------
    print("[DEBUG] Attribute inference stats:")
    print("   True unknown positives:", true_vals.sum())
    print("   Pred unknown positives:", preds.sum())
    print(f"   True positive rate: {true_vals.mean():.4f}")
    print(f"   Pred positive rate: {preds.mean():.4f}")
    corr = np.corrcoef(true_vals.flatten(), preds.flatten())[0,1]
    print(f"   Correlation (truth, pred): {corr:.4f}")

    return f1

# -----------------------------
# MEMBERSHIP INFERENCE (GPU)
# -----------------------------
def membership_inference_f1(real_train, real_test, synthetic):
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

    # -------------------------
    # DEBUG: print distance stats
    # -------------------------
    print("[DEBUG] Train distances: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        train_dists.min(), train_dists.max(), train_dists.mean()))
    print("[DEBUG] Test distances: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        test_dists.min(), test_dists.max(), test_dists.mean()))

    # -------------------------
    # Automatic threshold (5th percentile)
    # -------------------------
    combined = np.concatenate([train_dists, test_dists])
    threshold = np.percentile(combined, 5)  # top 5% closest are predicted as members
    print(f"[DEBUG] Auto-selected threshold: {threshold:.4f}")

    train_pred = (train_dists < threshold).astype(int)
    test_pred  = (test_dists  < threshold).astype(int)

    preds = np.concatenate([train_pred, test_pred])
    true  = np.concatenate([np.ones(len(train_pred)), np.zeros(len(test_pred))])

    # -------------------------
    # DEBUG: confusion stats
    # -------------------------
    print("[DEBUG] Membership predictions:")
    print("   True members:", true[:len(train_pred)].sum())
    print("   Pred members:", train_pred.sum())
    tp = ((train_pred == 1) & (true[:len(train_pred)] == 1)).sum()
    fp = ((train_pred == 1) & (true[:len(train_pred)] == 0)).sum()
    tn = ((train_pred == 0) & (true[:len(train_pred)] == 0)).sum()
    fn = ((train_pred == 0) & (true[:len(train_pred)] == 1)).sum()
    print(f"   TP={tp}, FP={fp}, TN={tn}, FN={fn}")

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
    SAMPLE_SIZE = 10000
    N_REPEATS = 5  # number of bootstraps to compute mean ± SD

    # Setup S3
    s3 = boto3.client("s3")

    # -----------------------------
    # Download the real and synthetic files
    # -----------------------------
    real_file = "original_vectors_gemma.npy"
    synthetic_file = "sample_output.npy"
    col_label_file = "patient_vector_columns.csv"

    full_real_data = download_s3_file(s3, BUCKET, real_file, f"/tmp/{real_file}")
    synthetic = download_s3_file(s3, BUCKET, synthetic_file, f"/tmp/{synthetic_file}")
    column_labels = download_column_labels(s3, BUCKET, col_label_file)

    # ---------------------------
    # Normalize age column
    # ---------------------------
    AGE_IDX = 1  # change to actual age column index
    full_real_data[:, AGE_IDX] = full_real_data[:, AGE_IDX] / 91.0

    log(f"Full real data shape: {full_real_data.shape}")
    log(f"Synthetic data shape: {synthetic.shape}")

    # -----------------------------
    # Compute high-entropy known features
    # -----------------------------
    p = full_real_data.mean(axis=0)
    entropy = -p * np.log2(p + 1e-12) - (1 - p) * np.log2(1 - p + 1e-12)
    known_idx = np.argsort(-entropy)[:256]  # top 256 high-entropy features
    unknown_idx = np.array([i for i in range(full_real_data.shape[1]) if i not in known_idx])

    # Map top high-entropy indices to column labels
    high_entropy_labels = [column_labels[i] for i in known_idx[:10]]
    log(f"Top 10 high-entropy feature labels: {high_entropy_labels}")

    # -----------------------------
    # Bootstrapping for mean ± SD
    # -----------------------------
    attr_f1_list = []
    memb_f1_list = []

    for i in range(N_REPEATS):
        # Sample
        indices = np.random.choice(full_real_data.shape[0], SAMPLE_SIZE, replace=False)
        sampled_real = full_real_data[indices]
        real_train, real_test = train_test_split(sampled_real, test_size=0.2, random_state=i)

        # Attribute inference
        attr_f1 = attribute_inference_f1(real_train, synthetic, known_idx, unknown_idx)
        attr_f1_list.append(attr_f1)

        # Membership inference
        memb_f1 = membership_inference_f1(real_train, real_test, synthetic)
        memb_f1_list.append(memb_f1)

    # Compute mean and SD
    attr_mean, attr_sd = np.mean(attr_f1_list), np.std(attr_f1_list)
    memb_mean, memb_sd = np.mean(memb_f1_list), np.std(memb_f1_list)

    log("---------- FINAL RESULTS (mean ± SD) ----------")
    log(f"Attribute Inference F1:   {attr_mean:.4f} ± {attr_sd:.4f}")
    log(f"Membership Inference F1: {memb_mean:.4f} ± {memb_sd:.4f}")

    return {
        "attribute_inference_f1_mean": float(attr_mean),
        "attribute_inference_f1_sd": float(attr_sd),
        "membership_inference_f1_mean": float(memb_mean),
        "membership_inference_f1_sd": float(memb_sd)
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
