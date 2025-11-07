import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
import modal
import tempfile
import boto3

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


# ---- Helper: Extract a score from model output ----
def parse_score(text):
    text = text.strip()
    if "10" in text:
        return 10
    digits = [int(c) for c in text if c.isdigit()]
    return digits[0] if digits else 5


# ---- Helper: sample model outputs (policy action) ----
@torch.no_grad()
def sample_scores(vectors, tokenizer, model, device):
    import re
    
    # Convert vectors → table CSV
    tables = [vector_to_table(v).to_csv(index=False) for v in vectors]

    # SAME PROMPT AS EVALUATION
    prompts = [
        (
            "You are a clinical data auditor.\n"
            "Given the synthetic patient record below, rate how realistic it is.\n"
            "Return ONLY a single integer from 1 to 10.\n\n"
            f"{tbl}\n\n"
            "Rating (1-10):"
        )
        for tbl in tables
    ]

    # Tokenize
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    input_ids = enc["input_ids"]
    input_lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)

    # Generate with sampling (RL exploration)
    gen_ids = model.generate(
        **enc,
        max_new_tokens=4,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Slice continuation tokens for each sample (Batched)
    continuations = [
        gen_ids[i][input_lengths[i]:]    # remove the prompt portion
        for i in range(len(gen_ids))
    ]

    # Decode + extract scores
    decoded = tokenizer.batch_decode(continuations, skip_special_tokens=True)

    scores = []
    for text in decoded:
        text = text.strip().replace(",", " ")
        m = re.search(r"\b(10|[1-9])\b", text)
        scores.append(int(m.group(1)) if m else 5)

    scores = torch.tensor(scores, dtype=torch.float32, device=device)

    return gen_ids, enc, scores



# ---- Helper: compute sequence log prob ----
# needed because RLVR performs a policy-gradient update, which is based on log-probs, not standard loss
def seq_logprob(model, input_ids, attn_mask, labels=None):
    """
    Compute sequence log-probabilities.
    If labels=None, compute log-probs for input_ids itself (causal LM).
    """
    if labels is None:
        labels = input_ids.clone()
    
    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    logits = outputs.logits
    token_mask = (labels != -100).float()

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * token_mask

    return token_logp.sum(dim=1)  # sum over tokens → shape [batch]


# ---- RLVR Trainer ----
class RLVRTrainer:
    def __init__(self, lr=1e-5, kl_beta=0.05, device="cuda"):
        self.device = device

        # --- Load directly from Hugging Face ---
        model_name = "RUCKBReasoning/TableLLM-8b"  # replace baseline_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.policy = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": "cuda"},   # or "auto" if you have multiple GPUs
            offload_folder="offload",
            dtype=torch.float16
        )
        self.ref = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": "cpu"},  # fully on CPU
            dtype=torch.float16
        ).eval()

        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.kl_beta = kl_beta

    def train_epoch(self, X_real, X_synth, batch=2, max_samples=256):
        # compute verifiable realism scores
        vectors = X_synth[:max_samples]
        vr_scores = compute_realism_scores(X_real, vectors)
        vr_scores = torch.tensor(vr_scores, dtype=torch.float32, device=self.device)

        avg_loss = 0
        n = 0

        for i in tqdm(range(0, len(vectors), batch), desc="RLVR Training"):
            batch_vectors = vectors[i:i+batch]
            batch_vr = vr_scores[i:i+batch]

            # --- 1) Sample new sequences for exploration ---
            gen_ids, enc_inputs, llm_scores = sample_scores(batch_vectors, self.tokenizer, self.policy, self.device)

            # --- 2) Compute log-probs on the original input_ids ---
            labels = enc_inputs.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            logp_policy = seq_logprob(self.policy, enc_inputs.input_ids, enc_inputs.attention_mask, labels)

            with torch.no_grad():
                logp_ref = seq_logprob(
                    self.ref,
                    enc_inputs.input_ids.cpu(),
                    enc_inputs.attention_mask.cpu(),
                    labels.cpu()
                ).to(self.device)

            # --- 3) Compute reward & advantage ---
            reward = 1 - (torch.abs(llm_scores - batch_vr) / 9.0)
            reward = reward.clamp(0, 1)
            advantage = reward - reward.mean()

            # --- 4) Policy + KL loss ---
            policy_loss = -(advantage * logp_policy).mean()
            kl_loss = (logp_policy - logp_ref).mean()
            loss = policy_loss + self.kl_beta * kl_loss

            self.opt.zero_grad()
            loss.backward()
            clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()

            avg_loss += loss.item() * len(batch_vectors)
            n += len(batch_vectors)
            torch.cuda.empty_cache()


        return avg_loss / n

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"\n[RLVR Model Saved] → {path}")


# ---------------------------
# Modal App
# ---------------------------
import modal
import os
import tempfile
import boto3
import numpy as np

app = modal.App("tablellm-evaluation")

@gpu_func := app.function(
    gpu="A100",
    secrets=[modal.Secret.from_name("aws-secret")],
    image=modal.Image.debian_slim().pip_install([
        "torch", "transformers", "scikit-learn", "pandas", "numpy", "tqdm", "boto3", "scipy", "accelerate"
    ]).run_commands(['export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True']),
    timeout=3600  # 1 hour max, adjust as needed
)
def run_rlvr(
    real_s3_key: str,          # S3 key for real data (.npy)
    synth_s3_key: str,         # S3 key for synthetic data (.npy)
    out_s3_prefix: str,        # S3 prefix to save RLVR model
    epochs: int = 3,
    lr: float = 1e-5,
    kl_beta: float = 0.05,
):
    tmp_root = tempfile.mkdtemp()
    print(f"📂 Working in temp dir: {tmp_root}")

    s3 = boto3.client("s3")

    # Parse bucket and key from real data path
    real_bucket, real_key = real_s3_key.split("/", 1)
    synth_bucket, synth_key = synth_s3_key.split("/", 1)
    out_bucket, out_prefix = out_s3_prefix.split("/", 1)

    # Local paths
    real_path = os.path.join(tmp_root, "real.npy")
    synth_path = os.path.join(tmp_root, "synth.npy")


    # --- Download real + synthetic data ---
    s3.download_file(real_bucket, real_key, real_path)
    s3.download_file(synth_bucket, synth_key, synth_path)
    print("✅ Downloaded real + synthetic data from S3")

    # --- Load trainer ---
    trainer = RLVRTrainer(lr=lr, kl_beta=kl_beta, device="cuda")

    # --- Load data ---
    X_real = np.load(real_path)
    X_synth = np.load(synth_path)

    # --- Training loop ---
    for epoch in range(epochs):
        loss = trainer.train_epoch(X_real, X_synth)
        print(f"Epoch {epoch+1} Loss = {loss:.4f}")

    # --- Save RLVR model locally ---
    out_dir = os.path.join(tmp_root, "rlvr_model")
    trainer.save(out_dir)

    # --- Upload RLVR model to S3 ---
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, out_dir)
            s3_key = os.path.join(out_prefix, rel_path)
            s3.upload_file(local_path, out_bucket, s3_key)
            print(f"✅ Uploaded {local_path} → s3://{out_bucket}/{s3_key}")

# ---------------------------
# Local entrypoint (optional)
# ---------------------------
@app.local_entrypoint()
def main():
    run_rlvr.remote(
        real_s3_key="health-forge-data-processing/ehr_norm.npy",
        synth_s3_key="health-forge-data-processing/workdirs/judge_train/samples/all_x.npy",
        out_s3_prefix="health-forge-data-processing/rlvr_judge",
        epochs=3,
        lr=1e-5,
        kl_beta=0.05
    )
