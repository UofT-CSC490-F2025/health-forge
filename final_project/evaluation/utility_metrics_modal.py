import modal
import boto3
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# -----------------------------
# Modal GPU image
# -----------------------------
image = (
    modal.Image.debian_slim()
    .pip_install("boto3", "numpy", "scikit-learn", "torch", "pandas", "matplotlib")
)

app = modal.App("ehr-utility-gpu")

def log(msg):
    print(f"[LOG] {msg}")

# -----------------------------
# Helpers
# -----------------------------
def download_s3_file(s3, bucket, key, local_path):
    log(f"Downloading s3://{bucket}/{key}")
    s3.download_file(bucket, key, local_path)
    return np.load(local_path)

def download_column_labels(s3, bucket, key):
    log(f"Downloading column labels from s3://{bucket}/{key}")
    local_path = f"/tmp/{os.path.basename(key)}"
    s3.download_file(bucket, key, local_path)
    df = pd.read_csv(local_path, header=None)
    return df.iloc[0].tolist()

# -----------------------------
# MAIN MODAL FUNCTION
# -----------------------------
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("aws-secret")],
    gpu="H100",
    timeout=60*60*2
)
def run():
    BUCKET = "healthforge-final-bucket"
    TRAIN_FILE = "original_vectors_train.npy"
    TEST_FILE  = "original_vectors_val.npy"
    SYNTH_FILE = "sample_output.npy"
    COL_LABEL_FILE = "patient_vector_columns.csv"
    N_REPEATS = 5
    SUBSAMPLE_FRAC = 0.01

    # Setup S3
    s3 = boto3.client("s3")

    # -----------------------------
    # Download train/test/synthetic
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

    log(f"Original real train shape: {real_train.shape}")
    log(f"Original real test shape: {real_test.shape}")
    log(f"Synthetic shape: {synthetic.shape}")

    # -----------------------------
    # Subsample
    # -----------------------------
    train_size = int(real_train.shape[0] * SUBSAMPLE_FRAC)
    test_size  = int(real_test.shape[0]  * SUBSAMPLE_FRAC)
    train_indices = np.random.choice(real_train.shape[0], train_size, replace=False)
    test_indices  = np.random.choice(real_test.shape[0],  test_size,  replace=False)
    real_train_sub = real_train[train_indices]
    real_test_sub  = real_test[test_indices]

    log(f"Subsampled real train shape: {real_train_sub.shape}")
    log(f"Subsampled real test shape: {real_test_sub.shape}")

    # -----------------------------
    # Combine full real dataset
    # -----------------------------
    full_real = np.vstack([real_train_sub, real_test_sub])

    # -----------------------------
    # Precompute entropies
    # -----------------------------
    p = full_real.mean(axis=0)
    p = np.clip(p, 0, 1)
    entropy = -p * np.log2(p + 1e-12) - (1 - p) * np.log2(1 - p + 1e-12)
    top_entropy_idx = np.argsort(-entropy)[:30]

    # -----------------------------
    # Bootstrapping
    # -----------------------------
    prevalence_corr_list = []
    nzc_list = []
    cmd_list = []
    f1_real_list = []
    f1_synth_list = []

    for i in range(N_REPEATS):
        log(f"Bootstrap repeat {i+1}/{N_REPEATS}")

        real_sample = full_real[np.random.choice(full_real.shape[0], full_real.shape[0], replace=True)]
        synth_sample = synthetic[np.random.choice(synthetic.shape[0], synthetic.shape[0], replace=True)]

        # Prevalence corr
        mean_real = real_sample.mean(axis=0)
        mean_synth = synth_sample.mean(axis=0)
        if np.std(mean_real) < 1e-12 or np.std(mean_synth) < 1e-12:
            corr = 0.0
        else:
            corr = np.corrcoef(mean_real, mean_synth)[0,1]
        prevalence_corr_list.append(corr)

        # NZC
        nzc = (synth_sample.sum(axis=0) > 0).sum()
        nzc_list.append(nzc)

        # CMD
        # Safe variance check in BOTH real + synthetic
        real_std = np.std(real_sample, axis=0)
        synth_std = np.std(synth_sample, axis=0)

        valid_cols = np.where((real_std > 1e-12) & (synth_std > 1e-12))[0]

        if len(valid_cols) < 2:
            # not enough variance → fallback to zeros
            real_corr = np.zeros((2,2))
            synth_corr = np.zeros((2,2))
        else:
            real_corr = np.corrcoef(real_sample[:, valid_cols], rowvar=False)
            synth_corr = np.corrcoef(synth_sample[:, valid_cols], rowvar=False)

        # Replace any remaining NaNs
        real_corr = np.nan_to_num(real_corr)
        synth_corr = np.nan_to_num(synth_corr)
        
        cmd = np.mean(np.abs(real_corr - synth_corr))
        cmd_list.append(cmd)

        # F1
        f1_real = []
        f1_synth = []

        for idx in top_entropy_idx:
            if idx not in binary_cols:
                continue
            y_real = full_real[:, idx]
            X_real = np.delete(full_real, idx, axis=1)
            y_synth = synth_sample[:, idx]
            X_synth = np.delete(synth_sample, idx, axis=1)

            # Skip constant columns
            if np.all(y_synth == 0) or np.all(y_synth == 1):
                continue
            if np.all(y_real == 0) or np.all(y_real == 1):
                continue

            y_real_bin = (y_real > 0.5).astype(int)
            y_synth_bin = (y_synth > 0.5).astype(int)

            clf_real = LogisticRegression(max_iter=1000)
            clf_real.fit(X_real, y_real_bin)
            f1_real.append(f1_score(y_real_bin, clf_real.predict(X_real), average="micro"))

            clf_synth = LogisticRegression(max_iter=1000)
            clf_synth.fit(X_synth, y_synth_bin)
            f1_synth.append(f1_score(y_synth_bin, clf_synth.predict(X_synth), average="micro"))

        f1_real_list.append(f1_real)
        f1_synth_list.append(f1_synth)

    # -----------------------------
    # Aggregate metrics
    # -----------------------------
    prevalence_corr_mean = np.mean(prevalence_corr_list)
    prevalence_corr_sd   = np.std(prevalence_corr_list)

    nzc_mean = np.mean(nzc_list)
    nzc_sd   = np.std(nzc_list)

    cmd_mean = np.mean(cmd_list)
    cmd_sd   = np.std(cmd_list)

    f1_real_array = np.array(f1_real_list)
    f1_synth_array = np.array(f1_synth_list)

    f1_real_mean = np.mean(f1_real_array, axis=0)
    f1_real_sd   = np.std(f1_real_array, axis=0)
    f1_synth_mean = np.mean(f1_synth_array, axis=0)
    f1_synth_sd   = np.std(f1_synth_array, axis=0)

    # -----------------------------
    # Compute correlations for plots
    # -----------------------------
    mean_real_full = full_real.mean(axis=0)
    mean_synth_full = synthetic.mean(axis=0)
    if np.std(mean_real_full) < 1e-12 or np.std(mean_synth_full) < 1e-12:
        full_prevalence_corr = 0.0
    else:
        full_prevalence_corr = np.corrcoef(mean_real_full, mean_synth_full)[0,1]


    if np.std(f1_real_mean) < 1e-12 or np.std(f1_synth_mean) < 1e-12:
        f1_corr = 0.0
    else:
        f1_corr = np.corrcoef(f1_real_mean, f1_synth_mean)[0,1]

    # -----------------------------
    # PLOTS WITH corr=... ANNOTATIONS
    # -----------------------------
    plt.figure(figsize=(6,6))
    plt.scatter(mean_real_full, mean_synth_full, alpha=0.3)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Real prevalence")
    plt.ylabel("Synthetic prevalence")
    plt.title("Feature prevalence (real vs synthetic)")

    plt.text(
        0.05, 0.90,
        f"corr = {full_prevalence_corr:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig("/tmp/dim_wise_prevalence.png")

    plt.figure(figsize=(6,6))
    plt.scatter(f1_real_mean, f1_synth_mean, alpha=0.7)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("F1 (real)")
    plt.ylabel("F1 (synthetic)")
    plt.title("Dimension-wise prediction F1")

    plt.text(
        0.05, 0.90,
        f"corr = {f1_corr:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig("/tmp/dim_wise_prediction.png")

    # Upload
    s3.upload_file("/tmp/dim_wise_prevalence.png", BUCKET, "dim_wise_prevalence.png")
    log(f"Uploaded /tmp/dim_wise_prevalence.png")
    s3.upload_file("/tmp/dim_wise_prediction.png", BUCKET, "dim_wise_prediction.png")
    log(f"Uploaded /tmp/dim_wise_prediction.png")

    # -----------------------------
    # Final logs
    # -----------------------------
    log("---------- FINAL RESULTS (mean ± SD) ----------")
    log(f"Dimension-wise prevalence correlation: {prevalence_corr_mean:.4f} ± {prevalence_corr_sd:.4f}")
    log(f"Non-zero code columns (NZC): {nzc_mean:.1f} ± {nzc_sd:.1f}")
    log(f"Correlation matrix distance (CMD): {cmd_mean:.4f} ± {cmd_sd:.4f}")
    log(f"Dimension-wise prediction F1 (real vs synthetic): {np.mean(f1_real_mean):.4f} ± {np.mean(f1_real_sd):.4f}")

    return {
        "prevalence_corr_mean": float(prevalence_corr_mean),
        "prevalence_corr_sd": float(prevalence_corr_sd),
        "nzc_mean": float(nzc_mean),
        "nzc_sd": float(nzc_sd),
        "cmd_mean": float(cmd_mean),
        "cmd_sd": float(cmd_sd),
        "f1_real_mean": float(np.mean(f1_real_mean)),
        "f1_real_sd": float(np.mean(f1_real_sd)),
        "f1_synth_mean": float(np.mean(f1_synth_mean)),
        "f1_synth_sd": float(np.mean(f1_synth_sd))
    }


# -----------------------------
# Auto-run when script executed
# -----------------------------
@app.local_entrypoint()
def main():
    run.remote()
    print("Submitted utility metrics job to Modal…")

if __name__ == "__main__":
    with app.run():
        main()
