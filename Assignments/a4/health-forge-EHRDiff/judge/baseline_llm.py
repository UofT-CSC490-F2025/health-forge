import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
import torch
from omegaconf import OmegaConf
from runners import generate_base
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# Metric Computation
def compute_realism_scores(X_real, X_synth, mah_weight=0.7, knn_weight=0.3):
    X_real = X_real[:, 1:]
    mu = np.mean(X_real, axis=0)
    cov = np.cov(X_real, rowvar=False)
    diff = X_synth - mu
    maha_dist = np.sqrt(np.sum((diff @ np.linalg.pinv(cov)) * diff, axis=1) + 1e-9)

    n_neighbors = max(1, int(0.1 * X_real.shape[0]))
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X_real)
    neigh_dists, _ = neigh.kneighbors(X_synth)
    neigh_dists = np.mean(neigh_dists, axis=1)

    score = - (mah_weight * maha_dist) + (knn_weight * neigh_dists)
    score = (score - score.min()) / (score.max() - score.min())
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


# Main Baseline Pipeline
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
        iterator = range(len(X_synth))
        if show_progress:
            iterator = tqdm(iterator, desc="Rating synthetic EHRs (TableLLM)")

        for i in iterator:
            df = vector_to_table(X_synth[i])
            table_str = df.to_csv(index=False)
            prompt = (
                "[INST]Rate from 1 to 10 how realistic this synthetic patient record looks, "
                "considering demographics and medical activity statistics.\n"
                "### [Table]\n"
                f"{table_str}"
                "[/INST]"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=5)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            digits = [int(s) for s in response if s.isdigit()]
            score = digits[0] if digits else 5
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
        
        sample_file = os.path.join(
        self.config.setup.root_dir,
        self.config.setup.workdir,
        "judge_train", 
        "samples",
        "all_x.npy"
    )
    
        # Ensure the file exists
        assert os.path.exists(sample_file), f"[ERROR] Synthetic sample file not found: {sample_file}"

        # Load and return the synthetic samples
        return np.load(sample_file)

    # Load Real Dataset
    def load_real(self):
        dataset_path = os.path.join(self.config.setup.root_dir, self.config.setup.dataset_dir)
        assert os.path.exists(dataset_path), "[ERROR] Real dataset not found"
        return np.load(dataset_path)

    # Run Evaluation
    def run(self):
        X_real = self.load_real()
        X_synth = self.generate_synthetic()

        # Only evaluate first 10 samples
        n_eval = 10
        X_synth_eval = X_synth[:n_eval]
        y_true = np.ones(n_eval, dtype=int)  # all synthetic

        # Baseline: score-based classifier
        baseline_scores = compute_realism_scores(X_real, X_synth_eval)
        baseline_threshold = 0.5
        baseline_pred = (baseline_scores > baseline_threshold).astype(int)

        # LLM: rate samples
        llm_scores = self.llm_rate_samples_tablellm(X_synth_eval, show_progress=True)
        llm_threshold = 5
        llm_pred = (llm_scores > llm_threshold).astype(int)

        # Quantitative Evaluation
        print("\n=== Baseline vs TableLLM Evaluation ===")
        print("\n-- Baseline Classifier --")
        print(f"Accuracy: {accuracy_score(y_true, baseline_pred):.3f}")
        print(f"Precision: {precision_score(y_true, baseline_pred, zero_division=0):.3f}")
        print(f"Recall: {recall_score(y_true, baseline_pred, zero_division=0):.3f}")
        print(f"F1: {f1_score(y_true, baseline_pred, zero_division=0):.3f}")

        print("\n-- TableLLM Classifier --")
        print(f"Accuracy: {accuracy_score(y_true, llm_pred):.3f}")
        print(f"Precision: {precision_score(y_true, llm_pred, zero_division=0):.3f}")
        print(f"Recall: {recall_score(y_true, llm_pred, zero_division=0):.3f}")
        print(f"F1: {f1_score(y_true, llm_pred, zero_division=0):.3f}")

        # Qualitative Analysis: disagreement examples
        disagree_idx = np.where(baseline_pred != llm_pred)[0]
        print(f"\nDisagreement examples (up to 5 shown, total {len(disagree_idx)}):")
        for i in disagree_idx[:5]:
            print(f"\nSample index: {i}")
            print(f"Baseline prediction: {baseline_pred[i]}, LLM prediction: {llm_pred[i]}, LLM score: {llm_scores[i]:.1f}")
            df = vector_to_table(X_synth_eval[i])
            print(df)

        # Save results
        save_dir = os.path.join(self.config.setup.root_dir, "baseline_tablellm")
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "baseline_scores.npy"), baseline_scores)
        np.save(os.path.join(save_dir, "llm_scores.npy"), llm_scores)
        print(f"\n[Baseline TableLLM Judge Completed] → Results saved in {save_dir}")



# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--root_dir", default=".")
    return parser.parse_args()


# Main
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
