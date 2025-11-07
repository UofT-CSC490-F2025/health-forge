import modal
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from omegaconf import OmegaConf
from tqdm import tqdm
import re
import os
import boto3
import tempfile



# Metric Computation
def compute_realism_scores(X_real, X_synth, mah_weight=0.85, knn_weight=0.15, neighbors=5):

    # --- MAHALANOBIS scoring (smaller = better) ---
    MAHA_BEST = 12   # anything ≤ 12 → score 10
    MAHA_WORST = 35.0  # anything ≥ 35 → score 1

    # --- kNN scoring (larger = better) ---
    KNN_WORST = 20.0  # collapse / too similar
    KNN_BEST = 40.0   # comfortably distinct but still in manifold


    Xr_raw = X_real[:, 1:]
    Xs_raw = X_synth

    # Standardize both using REAL distribution only
    scaler = StandardScaler().fit(Xr_raw)
    Xr = scaler.transform(Xr_raw)
    Xs = scaler.transform(Xs_raw)

    # Stable covariance estimate → Mahalanobis
    lw = LedoitWolf().fit(Xr)
    Sigma_inv = np.linalg.inv(lw.covariance_)
    diff_s = Xs - lw.location_
    maha_dist = np.sqrt(np.sum((diff_s @ Sigma_inv) * diff_s, axis=1))

    # kNN distances
    neigh = NearestNeighbors(n_neighbors=neighbors).fit(Xr)
    knn_dist = neigh.kneighbors(Xs, return_distance=True)[0].mean(axis=1)

    # Linear map to [0,1], where 1 = best, 0 = worst
    maha_t = (MAHA_WORST - maha_dist) / (MAHA_WORST - MAHA_BEST)
    maha_t = np.clip(maha_t, 0.0, 1.0)

    # Convert to 1–10
    maha_score = 1 + 9 * maha_t

    # Linear map to [0,1], where 1 = best, 0 = worst
    knn_t = (knn_dist - KNN_WORST) / (KNN_BEST - KNN_WORST)
    knn_t = np.clip(knn_t, 0.0, 1.0)

    # Convert to 1–10
    knn_score = 1 + 9 * knn_t

    # --- Combine ---
    score = mah_weight * maha_score + knn_weight * knn_score

    # Round to nearest integer in 1–10 range
    score = np.rint(score).astype(int)

    
    # === Print Summary ===
    print("\n=== Baseline Realism Score Summary (1–10) ===")
    print(f"Combined Score: mean={score.mean():.2f}, std={score.std():.2f}, min={score.min():.2f}, max={score.max():.2f}")
    print(f"Mahalanobis Subscore: mean={maha_score.mean():.2f}, std={maha_score.std():.2f}, min={maha_score.min():.2f}, max={maha_score.max():.2f}")
    print(f"kNN Subscore: mean={knn_score.mean():.2f}, std={knn_score.std():.2f}, min={knn_score.min():.2f}, max={knn_score.max():.2f}")

    return score


# Vector → Table Conversion
def vector_to_table(vector):
    table_splits = {
        "Prescriptions": 829,
        "Diagnoses": 1472,
        "Procedures": 352,
        "POE": 15,
        "Services": 13,
        "Admissions": 31,
        "OMR": 14,
        "HospitalEvents": 19,
        "Pharmacy": 584
    }

    data = {
        "SubjectID": int(vector[0]),
        "Gender": int(vector[1]),
        "Age": int(vector[2]),
        "IsDead": int(vector[3]),
    }

    offset = 4
    for name, length in table_splits.items():
        seg = np.array(vector[offset:offset + length], dtype=float)
        data[f"{name}_mean"] = np.mean(seg)
        data[f"{name}_sum"] = np.sum(seg)
        data[f"{name}_max"] = np.max(seg)
        data[f"{name}_min"] = np.min(seg)
        data[f"{name}_std"] = np.std(seg)
        offset += length

    return pd.DataFrame([data])

# ---------------------------
# Your Part3Pipeline + helpers go here
# ---------------------------
class Part3Pipeline:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("\n[Initializing TableLLM-8b model ...]")
        model_name = "RUCKBReasoning/TableLLM-8b"  # regular hyphen
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

        self.model.eval()
        print("[Model ready]\n")

    # LLM Table Scoring
    def llm_rate_samples_tablellm(self, X_synth, show_progress=False):
        scores = []
        iterator = tqdm(range(len(X_synth)), desc="Rating synthetic EHRs (TableLLM)") if show_progress else range(len(X_synth))

        for i in iterator:
            df = vector_to_table(X_synth[i])
            table_str = df.to_csv(index=False)

            prompt = (
                "You are a clinical data auditor.\n"
                "Given the synthetic patient record below, rate how realistic it is.\n"
                "Return ONLY a single integer from 1 to 10.\n\n"
                f"{table_str}\n\n"
                "Rating (1-10):"
            )

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)

            input_ids = inputs["input_ids"]
            input_length = input_ids.shape[1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                temperature=0.0,
            )

            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            print("Response:", response)  # Debug print

            match = re.search(r"\b([1-9]|10)\b", response)
            score = int(match.group(1)) if match else 5


            print("score:", score)  # Debug print

            scores.append(score)

        return np.array(scores)



    # Generate Synthetic Samples
    def generate_synthetic(self):

        # This part is commented out when using an existing synthetic data file
        '''
        workdir = os.path.join(self.config.setup.root_dir, self.config.setup.workdir)
        generate_base.evaluation(self.config, workdir)
        sample_file = os.path.join(workdir, "samples", "all_x.npy")
        assert os.path.exists(sample_file), "[ERROR] Failed to generate synthetic samples"
        return np.load(sample_file)
        '''
        
        s3_bucket = "health-forge-data-processing"
        s3_key = "workdirs/judge_train/samples/all_x.npy"
        local_path = "/tmp/all_x.npy"

        if not os.path.exists(local_path):
            s3 = boto3.client("s3")
            s3.download_file(s3_bucket, s3_key, local_path)

        return np.load(local_path)

    # Load Real Dataset
    def load_real(self):
        dataset_path = os.path.join(self.config.setup.root_dir, self.config.setup.dataset_dir)
        assert os.path.exists(dataset_path), "[ERROR] Real dataset not found"
        return np.load(dataset_path)

    # Run Evaluation
    def run(self):
        X_real = self.load_real()
        X_synth = self.generate_synthetic()

        # Only evaluate first 100 synthetic samples
        n_eval = 100
        X_synth_eval = X_synth[:n_eval]

        # Baseline realism score (1–10)
        baseline_scores = compute_realism_scores(X_real, X_synth_eval)

        # TableLLM realism score (1–10)
        llm_scores = self.llm_rate_samples_tablellm(X_synth_eval, show_progress=True)

        from scipy.stats import pearsonr, spearmanr

        print("\n=== Realism Score Comparison (Ordinal 1–10) ===")

        print("\n-- Baseline Realism Scores --")
        print(f"Mean: {baseline_scores.mean():.2f}")
        print(f"Std:  {baseline_scores.std():.2f}")

        print("\n-- TableLLM Realism Scores --")
        print(f"Mean: {llm_scores.mean():.2f}")
        print(f"Std:  {llm_scores.std():.2f}")

        # Correlation agreement
        pearson_corr, _ = pearsonr(baseline_scores, llm_scores)
        spearman_corr, _ = spearmanr(baseline_scores, llm_scores)

        print("\n-- Agreement Between Evaluators --")
        print(f"Pearson Correlation (linear similarity): {pearson_corr:.3f}")
        print(f"Spearman Correlation (rank similarity):  {spearman_corr:.3f}")

        # Largest realism disagreements
        differences = np.abs(baseline_scores - llm_scores)
        sorted_idx = np.argsort(-differences)

        print(f"\n=== Top Realism Disagreement Examples (largest score gaps) ===")
        for i in sorted_idx[:5]:
            print(f"\nSample index: {i}")
            print(f"Baseline realism score: {baseline_scores[i]}")
            print(f"TableLLM realism score: {llm_scores[i]}")
            print(f"Difference: {differences[i]}")
            df = vector_to_table(X_synth_eval[i])
            print(df)

        # SAVE MODEL (this is what you need for RLVR)
        save_dir = os.path.join(self.config.setup.root_dir, "baseline_tablellm")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        print(f"\n✅ Baseline Model Saved → {save_dir}")

        # --- Upload to S3 ---
        s3_bucket = "health-forge-data-processing"
        s3_prefix = "baseline_tablellm"
        s3 = boto3.client("s3")
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, save_dir)
                s3_key = os.path.join(s3_prefix, relative_path)
                s3.upload_file(local_path, s3_bucket, s3_key)
                print(f"✅ Uploaded {local_path} → s3://{s3_bucket}/{s3_key}")


# ---------------------------
# Modal App
# ---------------------------
app = modal.App("tablellm-evaluation")

@gpu_func := app.function(
    gpu="A100",
    timeout=7200,
    secrets=[modal.Secret.from_name("aws-secret")],
    image=modal.Image.debian_slim().pip_install([
        "torch", "transformers", "scikit-learn", "pandas", "numpy", "tqdm", "omegaconf", "boto3", "scipy", "accelerate"
    ]),
)
def run_baseline_llm(
    s3_bucket: str,
    dataset_key: str,
    config_key: str,
    workdir_key: str = None,
):
    # --- Setup ---
    s3 = boto3.client("s3")
    tmp_root = tempfile.mkdtemp()
    print(f"📂 Working in temporary dir: {tmp_root}")

    # --- Download files from S3 ---
    dataset_path = os.path.join(tmp_root, "ehr_norm.npy")
    config_path = os.path.join(tmp_root, "train_cfg.yaml")
    workdir_path = os.path.join(tmp_root, "workdir")
    os.makedirs(workdir_path, exist_ok=True)

    s3.download_file(s3_bucket, dataset_key, dataset_path)
    s3.download_file(s3_bucket, config_key, config_path)
    print(f"✅ Downloaded dataset + config from S3")

    if workdir_key:
        s3.download_file(s3_bucket, workdir_key, os.path.join(workdir_path, "all_x.npy"))
        print(f"✅ Downloaded synthetic workdir sample file")

    # --- Load config ---
    config = OmegaConf.load(config_path)
    config.setup.dataset_dir = dataset_path
    config.setup.workdir = workdir_path
    config.setup.root_dir = tmp_root

    # --- Run baseline pipeline ---
    pipeline = Part3Pipeline(config)
    pipeline.run()
# ---------------------------
# Local entrypoint (optional)
# ---------------------------
@app.local_entrypoint()
def main():
    run_baseline_llm.remote(
        workdir="workdirs",
        dataset_dir="data/ehr_norm.npy",
        model_cfg="configs/cinc/cfg/train_cfg.yaml"
    )
