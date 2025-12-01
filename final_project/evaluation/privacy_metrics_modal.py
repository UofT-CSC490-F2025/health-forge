import modal
import boto3
import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

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
# ATTRIBUTE INFERENCE (CPU) with bootstrap & k-voting
# -----------------------------
def attribute_inference_f1(real_train, synthetic, known_idx, unknown_idx, k=5, n_bootstrap=5):
    f1_list = []

    for b in range(n_bootstrap):
        # Subsample synthetic rows for this bootstrap
        syn_sample = synthetic[np.random.choice(synthetic.shape[0], real_train.shape[0], replace=True)]
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(syn_sample[:, known_idx])

        _, indices = knn.kneighbors(real_train[:, known_idx])
        preds = []

        for i in range(real_train.shape[0]):
            neigh = indices[i]
            vote = syn_sample[neigh][:, unknown_idx].mean(axis=0) > 0.5
            preds.append(vote.astype(int))

        preds = np.array(preds)
        true_vals = real_train[:, unknown_idx]
        f1 = f1_score(true_vals.flatten(), preds.flatten(), zero_division=0, average="micro")
        f1_list.append(f1)

    mean_f1 = np.mean(f1_list)
    sd_f1 = np.std(f1_list)
    return mean_f1, sd_f1

# -----------------------------
# MEMBERSHIP INFERENCE (GPU) with bootstrap & batching
# -----------------------------
def membership_inference_f1(real_train, real_test, synthetic, n_bootstrap=5, max_synth_rows=2000):
    f1_list = []

    def gpu_min_distances(real, synthetic_batch, batch_size=500):
        real_t = torch.tensor(real, dtype=torch.float32, device="cuda")
        syn_t  = torch.tensor(synthetic_batch, dtype=torch.float32, device="cuda")
        mins = []

        for i in range(0, real_t.size(0), batch_size):
            batch_real = real_t[i:i+batch_size][:, None, :]
            dists = torch.norm(batch_real - syn_t[None, :, :], dim=2)
            batch_mins = dists.min(dim=1).values.cpu().numpy()
            mins.append(batch_mins)

        return np.concatenate(mins)

    for b in range(n_bootstrap):
        n_rows = min(max_synth_rows, synthetic.shape[0])
        syn_sample = synthetic[np.random.choice(synthetic.shape[0], n_rows, replace=True)]

        train_dists = gpu_min_distances(real_train, syn_sample)
        test_dists  = gpu_min_distances(real_test, syn_sample)

        combined = np.concatenate([train_dists, test_dists])
        threshold = np.percentile(combined, 5)

        train_pred = (train_dists < threshold).astype(int)
        test_pred  = (test_dists  < threshold).astype(int)
        preds = np.concatenate([train_pred, test_pred])
        true  = np.concatenate([np.ones(len(train_pred)), np.zeros(len(test_pred))])

        f1 = f1_score(true, preds)
        f1_list.append(f1)

    mean_f1 = np.mean(f1_list)
    sd_f1 = np.std(f1_list)
    return mean_f1, sd_f1

# -----------------------------
# MAIN MODAL FUNCTION
# -----------------------------
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("aws-secret")],
    gpu="H100",
    timeout=60 * 60 * 2,
)
def run():
    BUCKET = "healthforge-final-bucket-1"
    TRAIN_FILE = "original_vectors_train.npy"
    TEST_FILE  = "original_vectors_val.npy"
    SYNTH_FILE = "sample_output.npy"
    COL_LABEL_FILE = "patient_vector_columns.csv"
    N_REPEATS = 5

    # Setup S3
    s3 = boto3.client("s3")

    # -----------------------------
    # Download train/test/synthetic files
    # -----------------------------
    real_train = download_s3_file(s3, BUCKET, TRAIN_FILE, f"/tmp/{TRAIN_FILE}")
    real_test  = download_s3_file(s3, BUCKET, TEST_FILE,  f"/tmp/{TEST_FILE}")
    synthetic  = download_s3_file(s3, BUCKET, SYNTH_FILE, f"/tmp/{SYNTH_FILE}")
    column_labels = download_column_labels(s3, BUCKET, COL_LABEL_FILE)

    # -----------------------------
    # Normalize continuous columns
    # -----------------------------
    AGE_IDX = 1
    ADM_IDX = 3

    # Normalize by max value
    real_train[:, AGE_IDX] = real_train[:, AGE_IDX] / 91.0
    real_test[:, AGE_IDX]  = real_test[:, AGE_IDX]  / 91.0

    real_train[:, ADM_IDX] = real_train[:, ADM_IDX] / 238.0
    real_test[:, ADM_IDX]  = real_test[:, ADM_IDX]  / 238.0

    # -----------------------------
    # Identify binary columns for F1
    # -----------------------------
    # A column is binary if it contains only 0/1 in the real train set
    binary_cols = [i for i in range(real_train.shape[1]) if set(np.unique(real_train[:, i])) <= {0, 1}]

    log(f"Real train shape: {real_train.shape}")
    log(f"Real test shape: {real_test.shape}")
    log(f"Synthetic shape: {synthetic.shape}")

    # Subsample 1/10 of the train and test sets
    train_size = real_train.shape[0] // 100
    test_size  = real_test.shape[0]  // 100

    train_indices = np.random.choice(real_train.shape[0], train_size, replace=False)
    test_indices  = np.random.choice(real_test.shape[0],  test_size,  replace=False)

    real_train = real_train[train_indices]
    real_test  = real_test[test_indices]

    log(f"Subsampled real train shape: {real_train.shape}")
    log(f"Subsampled real test shape: {real_test.shape}")


    # -----------------------------
    # Compute high-entropy known features
    # -----------------------------
    p = real_train[:, binary_cols].mean(axis=0)
    p = real_train.mean(axis=0)
    p = np.clip(p, 0, 1)
    entropy = -p * np.log2(p + 1e-12) - (1 - p) * np.log2(1 - p + 1e-12)
    known_idx = np.argsort(-entropy)[:256]  # top 256 high-entropy features
    binary_unknown_idx = [i for i in range(real_train.shape[1]) if i not in known_idx and i in binary_cols]

    # Map top high-entropy indices to column labels
    high_entropy_labels = [column_labels[i] for i in known_idx[:10]]
    log(f"Top 10 high-entropy feature labels: {high_entropy_labels}")

    # -----------------------------
    # Bootstrapping for mean ± SD
    # -----------------------------
    attr_f1_list = []
    memb_f1_list = []

    for i in range(N_REPEATS):
        log(f"Bootstrap repeat {i+1}/{N_REPEATS}")
        attr_f1 = attribute_inference_f1(real_train, synthetic, known_idx, binary_unknown_idx)
        memb_f1 = membership_inference_f1(real_train, real_test, synthetic)
        attr_f1_list.append(attr_f1)
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
