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
def attribute_inference_risk(real_train, synthetic, known_idx, unknown_idx, k=500, n_bootstrap=5):
    risk_list = []

    for b in range(n_bootstrap):
        # Subsample synthetic rows for this bootstrap
        syn_sample = synthetic[np.random.choice(synthetic.shape[0], real_train.shape[0], replace=True)]
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(syn_sample[:, known_idx])

        _, indices = knn.kneighbors(real_train[:, known_idx])
        preds = []

        for i in range(real_train.shape[0]):
            neigh = indices[i]
            # Predict each unknown binary feature using majority vote
            vote_prob = syn_sample[neigh][:, unknown_idx].mean(axis=0)
            preds.append(vote_prob)

        preds = np.array(preds)
        true_vals = real_train[:, unknown_idx]
        correct = (preds == true_vals).sum()
        total = np.prod(true_vals.shape)
        risk = correct / total
        risk_list.append(risk)


    mean_risk = np.mean(risk_list)
    sd_risk   = np.std(risk_list)
    return mean_risk, sd_risk


# -----------------------------
# MEMBERSHIP INFERENCE (GPU) with bootstrap & batching
# -----------------------------
# -----------------------------
# MEMBERSHIP INFERENCE (GPU) as risk
# -----------------------------
def membership_inference_risk(real_train, real_test, synthetic, n_bootstrap=5, max_synth_rows=10000):
    risk_list = []

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
        threshold = np.percentile(combined, 5)  # 5th percentile as membership threshold

        # Predict membership: 1 if distance < threshold, else 0
        train_pred = (train_dists < threshold).astype(int)
        test_pred  = (test_dists  < threshold).astype(int)
        preds = np.concatenate([train_pred, test_pred])

        # True membership labels: 1 for train, 0 for test
        true_labels = np.concatenate([np.ones(len(train_pred)), np.zeros(len(test_pred))])

        # Membership risk = fraction of correct predictions
        risk = (preds == true_labels).mean()
        risk_list.append(risk)

    mean_risk = np.mean(risk_list)
    sd_risk   = np.std(risk_list)
    return mean_risk, sd_risk


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
    SYNTH_FILE = "all_samples_unguided.npy"
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
    synthetic = synthetic.reshape(-1, synthetic.shape[-1])
    log(f"Flattened synthetic shape: {synthetic.shape}")
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
    train_size = real_train.shape[0] // 10
    test_size  = real_test.shape[0]  // 10

    train_indices = np.random.choice(real_train.shape[0], train_size, replace=False)
    test_indices  = np.random.choice(real_test.shape[0],  test_size,  replace=False)

    real_train = real_train[train_indices]
    real_test  = real_test[test_indices]

    log(f"Subsampled real train shape: {real_train.shape}")
    log(f"Subsampled real test shape: {real_test.shape}")


    # -----------------------------
    # Compute top-entropy known features (instead of top-frequency)
    # -----------------------------
    # Compute probability per column
    p = real_train[:, binary_cols].mean(axis=0)
    p = np.clip(p, 1e-12, 1-1e-12)  # avoid log(0)
    # Compute binary entropy
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    # Select top 30 highest-entropy features
    top_entropy_indices_in_binary = np.argsort(-entropy)[:30]
    known_idx = [binary_cols[i] for i in top_entropy_indices_in_binary]  # map back to original column indices

    # Remaining binary columns
    binary_unknown_idx = [i for i in binary_cols if i not in known_idx]

    # Map top known indices to labels
    top_entropy_labels = [column_labels[i] for i in known_idx[:10]]
    log(f"Top 10 high-entropy feature labels: {top_entropy_labels}")

    # -----------------------------
    # Bootstrapping for mean ± SD
    # -----------------------------
    attr_f1_list = []
    memb_f1_list = []

    for i in range(N_REPEATS):
        log(f"Bootstrap repeat {i+1}/{N_REPEATS}")
        attr_risk = attribute_inference_risk(real_train, synthetic, known_idx, binary_unknown_idx)
        attr_f1_list.append(attr_risk)
        memb_risk = membership_inference_risk(real_train, real_test, synthetic)
        memb_f1_list.append(memb_risk)


    # Compute mean and SD
    attr_mean, attr_sd = np.mean(attr_f1_list), np.std(attr_f1_list)
    memb_mean, memb_sd = np.mean(memb_f1_list), np.std(memb_f1_list)

    log("---------- FINAL RESULTS (mean ± SD) ----------")
    log(f"Attribute Inference Risk: {attr_mean:.4f} ± {attr_sd:.4f}")
    log(f"Membership Inference Risk: {memb_mean:.4f} ± {memb_sd:.4f}")

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
