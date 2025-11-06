import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr
from baseline_llm import vector_to_text, compute_realism_scores


def llm_rate(texts, tokenizer, model, device):
    scores = []
    for text in tqdm(texts, desc="Judging"):
        # Updated to match baseline prompt wording
        prompt = (
        "[INST]Rate from 1 to 10 how realistic this synthetic patient record looks, "
        "considering demographics and medical activity statistics.\n"
        "### [Table]\n"
        f"{text}"
        "[/INST]"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        output = model.generate(**inputs, max_new_tokens=5)
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        digits = [int(c) for c in pred if c.isdigit()]
        score = digits[0] if digits else 5
        scores.append(score)
    return np.array(scores)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_dir", required=True, help="Path to baseline or RLVR judge")
    parser.add_argument("--real_data", required=True)
    parser.add_argument("--synth_data", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.judge_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.judge_dir).to(device)

    X_real = np.load(args.real_data)
    X_synth = np.load(args.synth_data)

    # Compute realism scores (0–1 → convert to 1–10)
    vr = compute_realism_scores(X_real, X_synth)
    vr = 1 + 9 * vr

    # Convert synth vectors to text
    texts = [vector_to_text(x) for x in X_synth[:args.max_samples]]
    llm_scores = llm_rate(texts, tokenizer, model, device)

    # Trim VR to match predicted size
    vr = vr[:len(llm_scores)]

    # === Regression Metrics ===
    mae = mean_absolute_error(vr, llm_scores)
    corr = pearsonr(llm_scores, vr)[0]

    # === Classification Metrics ===
    # threshold realism: >5 means realistic, <=5 unrealistic
    y_true = (vr > 5).astype(int)
    y_pred = (llm_scores > 5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n=== Judge Evaluation (Regression) ===")
    print(f"MAE:   {mae:.3f}")
    print(f"Corr:  {corr:.3f}")

    print("\n=== Judge Evaluation (Classification) ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")


if __name__ == "__main__":
    main()