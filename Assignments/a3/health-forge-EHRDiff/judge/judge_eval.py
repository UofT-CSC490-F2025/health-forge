# judge_eval.py
# Evaluate the trained Judge model on held-out test data

import os, json, torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from judge_sft_train import ScalarJudge

# -------------------------
# Config
# -------------------------
CFG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "model_dir": "judge_sft",   
    "test_file": "rewards_data/test.jsonl",
    "max_seq_len": 1536,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# -------------------------
# Load model and tokenizer
# -------------------------
print(f"Using device: {CFG['device']}")
tok = AutoTokenizer.from_pretrained(CFG["model_dir"])
model = ScalarJudge(CFG["model_name"])
model.load_state_dict(torch.load(os.path.join(CFG["model_dir"], "pytorch_model.bin"),
                                 map_location=CFG["device"]))
model.eval().to(CFG["device"])

# -------------------------
# Load test data
# -------------------------
with open(CFG["test_file"], "r") as f:
    rows = [json.loads(l) for l in f]

preds, targets = [], []

# -------------------------
# Inference
# -------------------------
print(f"Evaluating on {len(rows)} test samples...")

for row in rows:
    ehr_json = row.get("ehr", {"ehr_id": row.get("ehr_id")})
    text = "EHR: " + json.dumps(ehr_json, separators=(",", ":"))
    enc = tok(text, truncation=True, max_length=CFG["max_seq_len"],
              return_tensors="pt").to(CFG["device"])

    with torch.no_grad():
        pred = model(**enc).item()

    preds.append(pred)
    targets.append(row["reward"])

# -------------------------
# Metrics
# -------------------------
preds = np.array(preds)
targets = np.array(targets)
mse = np.mean((preds - targets) ** 2)
rmse = np.sqrt(mse)
corr = np.corrcoef(preds, targets)[0, 1]

print("\n--- Test Results ---")
print(f"Test MSE  : {mse:.6f}")
print(f"Test RMSE : {rmse:.6f}")
print(f"Correlation (r): {corr:.4f}")

# -------------------------
# Visualization
# -------------------------
plt.figure(figsize=(6,6))
plt.scatter(targets, preds, alpha=0.6)
plt.plot([0,1], [0,1], "--", color="red")
plt.xlabel("True Reward (analytical)")
plt.ylabel("Predicted Reward (Judge)")
plt.title("Judge Predictions vs. True Rewards")
plt.grid(True)
plt.tight_layout()
plt.show()
