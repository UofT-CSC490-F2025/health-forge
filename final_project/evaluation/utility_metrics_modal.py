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
    SAMPLE_SIZE = 10000
    N_REPEATS = 5

    # Setup S3
    s3 = boto3.client("s3")
    real_file = "original_vectors_gemma.npy"
    synthetic_file = "sample_output.npy"
    col_label_file = "patient_vector_columns.csv"

    full_real_data = download_s3_file(s3, BUCKET, real_file, f"/tmp/{real_file}")
    synthetic = download_s3_file(s3, BUCKET, synthetic_file, f"/tmp/{synthetic_file}")
    column_labels = download_column_labels(s3, BUCKET, col_label_file)

    # Normalize age column
    AGE_IDX = 1
    full_real_data[:, AGE_IDX] = full_real_data[:, AGE_IDX] / 91.0

    log(f"Full real data shape: {full_real_data.shape}")
    log(f"Synthetic data shape: {synthetic.shape}")

    # -----------------------------
    # Precompute entropies
    # -----------------------------
    p = full_real_data.mean(axis=0)
    entropy = -p * np.log2(p + 1e-12) - (1 - p) * np.log2(1 - p + 1e-12)
    top_entropy_idx = np.argsort(-entropy)[:30]

    # -----------------------------
    # Bootstrapping
    # -----------------------------
    prevalence_corr_list = []
    nzc_list = []
    cmd_list = []
    f1_list = []

    for i in range(N_REPEATS):
        # Sample real data
        indices = np.random.choice(full_real_data.shape[0], SAMPLE_SIZE, replace=False)
        sampled_real = full_real_data[indices]

        # Split for prediction tasks
        split = int(SAMPLE_SIZE*0.8)
        real_train = sampled_real[:split]
        real_test = sampled_real[split:]

        # --------- Dimension-wise prevalence ---------
        mean_real = real_train.mean(axis=0)
        mean_synth = synthetic.mean(axis=0)
        corr = np.corrcoef(mean_real, mean_synth)[0,1]
        prevalence_corr_list.append(corr)

        # Non-zero code columns
        nzc = (synthetic.sum(axis=0) > 0).sum()
        nzc_list.append(nzc)

        # --------- Correlation Matrix Distance (CMD) ---------
        valid_cols = np.where(np.std(real_train, axis=0) > 1e-12)[0]
        real_corr = np.corrcoef(real_train[:, valid_cols], rowvar=False)
        synth_corr = np.corrcoef(synthetic[:, valid_cols], rowvar=False)
        cmd = np.mean(np.abs(real_corr - synth_corr))
        cmd_list.append(cmd)

        # --------- Dimension-wise prediction F1 ---------
        f1_tasks = []
        for idx in top_entropy_idx:
            y_real = real_test[:, idx]
            X_real = np.delete(real_test, idx, axis=1)
            X_synth = np.delete(synthetic, idx, axis=1)
            y_synth = synthetic[:, idx]

            # Skip constant targets
            if np.all(y_synth == 0) or np.all(y_synth == 1):
                continue
            if np.all(y_real == 0) or np.all(y_real == 1):
                continue

            # Binarize targets if needed
            y_synth_bin = (y_synth > 0.5).astype(int)
            y_real_bin = (y_real > 0.5).astype(int)

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_synth, y_synth_bin)
            y_pred = clf.predict(X_real)
            f1 = f1_score(y_real_bin, y_pred, average="micro")
            f1_tasks.append(f1)
        f1_list.append(f1_tasks)

    # --------- Aggregate metrics ---------
    prevalence_corr_mean = np.mean(prevalence_corr_list)
    prevalence_corr_sd = np.std(prevalence_corr_list)

    nzc_mean = np.mean(nzc_list)
    nzc_sd = np.std(nzc_list)

    cmd_mean = np.mean(cmd_list)
    cmd_sd = np.std(cmd_list)

    f1_array = np.array(f1_list)
    f1_mean = np.mean(f1_array, axis=0)
    f1_sd = np.std(f1_array, axis=0)

    # --------- Plots ---------
    plt.figure(figsize=(6,6))
    plt.scatter(mean_real, mean_synth, alpha=0.3)
    plt.plot([0,1],[0,1], color="red", linestyle="--")
    plt.xlabel("Real prevalence")
    plt.ylabel("Synthetic prevalence")
    plt.title("Dimension-wise probability scatter plot")
    plt.tight_layout()
    plt.savefig("/tmp/dim_wise_prevalence.png")

    plt.figure(figsize=(6,6))
    plt.errorbar(range(len(f1_mean)), f1_mean, yerr=f1_sd, fmt='o', alpha=0.7)
    plt.plot([0,29],[0,1], color="red", linestyle="--")
    plt.xlabel("Task index")
    plt.ylabel("Micro F1")
    plt.title("Dimension-wise prediction scatter plot")
    plt.tight_layout()
    plt.savefig("/tmp/dim_wise_prediction.png")

    # Upload dimension-wise prevalence plot
    prevalence_path = "/tmp/dim_wise_prevalence.png"
    s3.upload_file(prevalence_path, BUCKET, "dim_wise_prevalence.png")
    log(f"Uploaded {prevalence_path} to s3://{BUCKET}/dim_wise_prevalence.png")

    # Upload dimension-wise prediction plot
    prediction_path = "/tmp/dim_wise_prediction.png"
    s3.upload_file(prediction_path, BUCKET, "dim_wise_prediction.png")
    log(f"Uploaded {prediction_path} to s3://{BUCKET}/dim_wise_prediction.png")

    # --------- Logging ---------
    log("---------- FINAL RESULTS (mean ± SD) ----------")
    log(f"Dimension-wise prevalence correlation: {prevalence_corr_mean:.4f} ± {prevalence_corr_sd:.4f}")
    log(f"Non-zero code columns (NZC): {nzc_mean:.1f} ± {nzc_sd:.1f}")
    log(f"Correlation matrix distance (CMD): {cmd_mean:.4f} ± {cmd_sd:.4f}")
    log(f"Dimension-wise prediction micro F1: {np.mean(f1_mean):.4f} ± {np.mean(f1_sd):.4f}")

    return {
        "prevalence_corr_mean": float(prevalence_corr_mean),
        "prevalence_corr_sd": float(prevalence_corr_sd),
        "nzc_mean": float(nzc_mean),
        "nzc_sd": float(nzc_sd),
        "cmd_mean": float(cmd_mean),
        "cmd_sd": float(cmd_sd),
        "f1_mean": float(np.mean(f1_mean)),
        "f1_sd": float(np.mean(f1_sd))
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
