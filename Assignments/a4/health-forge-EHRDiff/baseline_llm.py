import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from omegaconf import OmegaConf
from runners import generate_base
from tqdm import tqdm

# Judge-like metric calculation
def compute_realism_scores(X_real, X_synth, mah_weight=0.7, knn_weight=0.3):
   
    # Mahalanobis distance
    mu = np.mean(X_real, axis=0)
    cov = np.cov(X_real, rowvar=False)
    diff = X_synth - mu
    maha_dist = np.sqrt(np.sum((diff @ np.linalg.pinv(cov)) * diff, axis=1) + 1e-9)

    # Distance to nearest neighbors
    n_neighbors = max(1, int(0.1 * X_real.shape[0]))
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X_real)
    neigh_dists, _ = neigh.kneighbors(X_synth)
    neigh_dists = np.mean(neigh_dists, axis=1)

    # Weighted difference
    score = - (mah_weight * maha_dist) + (knn_weight * neigh_dists)
    # Normalize 0–1 for comparison
    score = (score - score.min()) / (score.max() - score.min())
    return score

# Convert EHR vector to text
def vector_to_text(vector):
    text = f"Gender: {vector[0]}, Age: {vector[1]}\n"
    offset = 2
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
    for table_name, length in table_splits.items():
        vec_slice = vector[offset:offset + length]
        text += f"{table_name}: {vec_slice.tolist()[:10]} ...\n"
        offset += length
    return text

# LLM scoring
def llm_rate_samples(text_list, tokenizer, model, device, batch_size=8, show_progress=False):
    scores = []
    iterator = range(0, len(text_list), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Rating synthetic EHRs")
    for i in iterator:
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(
            [f"Rate from 1 to 10 how realistic this synthetic patient record looks:\n{text}" for text in batch],
            return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(device)
        outputs = model.generate(**inputs, max_new_tokens=5)
        for out in outputs:
            pred_text = tokenizer.decode(out, skip_special_tokens=True).strip()
            digits = [int(s) for s in pred_text if s.isdigit()]
            score = digits[0] if digits else 5
            scores.append(score)
    return np.array(scores)

# Main workflow
class Part3Pipeline:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(self.device)

    def generate_synthetic(self):
        workdir = os.path.join(self.config.setup.root_dir, self.config.setup.workdir)
        generate_base.evaluation(self.config, workdir)
        sample_file = os.path.join(workdir, "samples", "all_x.npy")
        assert os.path.exists(sample_file), "[ERROR] Failed to generate synthetic samples"
        return np.load(sample_file)

    def load_real(self):
        dataset_path = os.path.join(self.config.setup.root_dir, self.config.setup.dataset_dir)
        assert os.path.exists(dataset_path), "[ERROR] Real dataset not found"
        return np.load(dataset_path)

    def run(self):
        # Load data
        X_real = self.load_real()
        X_synth = self.generate_synthetic()

        # Compute baseline realism metric
        baseline_scores = compute_realism_scores(X_real, X_synth)
        baseline_pred = (baseline_scores < 0.5).astype(int)

        # Convert synthetic EHR to text
        X_synth_texts = [vector_to_text(x) for x in X_synth[:100]]  # sample 100 for demo

        # LLM scoring
        llm_scores = llm_rate_samples(X_synth_texts, self.tokenizer, self.model, self.device, show_progress=True)
        llm_pred = (llm_scores > 5).astype(int)

        # Compare baseline vs LLM
        y_true = llm_pred
        y_pred_baseline = baseline_pred[:len(y_true)]

        print("\n=== Baseline vs LLM Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_true, y_pred_baseline):.3f}")
        print(f"Precision: {precision_score(y_true, y_pred_baseline, zero_division=0):.3f}")
        print(f"Recall: {recall_score(y_true, y_pred_baseline, zero_division=0):.3f}")
        print(f"F1: {f1_score(y_true, y_pred_baseline, zero_division=0):.3f}")

        # Qualitative error analysis
        errors = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred_baseline[i]:
                errors.append({
                    "true_label": int(y_true[i]),
                    "baseline_pred": int(y_pred_baseline[i]),
                    "llm_score": int(llm_scores[i]),
                    "text_sample": X_synth_texts[i][:200] + "..."
                })

        print(f"\nTotal error cases: {len(errors)}")
        for e in errors[:5]:
            print(f"\nTrue: {e['true_label']} | Baseline: {e['baseline_pred']} | LLM Score: {e['llm_score']}")
            print("Sample:", e["text_sample"])

        # === Save Baseline LLM Judge ===
        save_dir = os.path.join(self.config.setup.root_dir, "baseline")
        os.makedirs(save_dir, exist_ok=True)

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        print(f"\n[Baseline Judge Saved] → {save_dir}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--root_dir", default=".")
    return parser.parse_args()

def main():
    args = parse_args()
    config = OmegaConf.load(args.model_cfg)
    config.setup.workdir = args.workdir
    config.setup.dataset_dir = args.dataset_dir
    config.setup.root_dir = args.root_dir

    pipeline = Part3Pipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()