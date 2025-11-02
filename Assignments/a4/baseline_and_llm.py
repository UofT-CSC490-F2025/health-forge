import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# Load dataset
data = np.load("ehr_norm.npy")
print("Loaded data shape:", data.shape)

# Column mapping:
# 0 - Subject ID, 1 - Gender, 2 - Age, 3 - isDead, 4... - EHR tables
y = data[:, 3].astype(int)  # label
X = np.delete(data[:, 1:], 2, axis=1)  # remove isDead from features

# Train / Validation / Test split (70/15/15)
X_train_full, X_temp, y_train_full, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val_num, X_test_num, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train_full.shape}, Val: {X_val_num.shape}, Test: {X_test_num.shape}")

# Logistic Regression Hyperparameter Search
print("\n=== Hyperparameter Search on Validation Set ===")
best_f1 = 0
best_model = None
best_params = None

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["lbfgs"]
}

for C in param_grid["C"]:
    model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000)
    model.fit(X_train_full, y_train_full)
    y_pred_val = model.predict(X_val_num)
    f1 = f1_score(y_val, y_pred_val, zero_division=0)
    print(f"C={C:<6} | Val F1={f1:.3f}")
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_params = {"C": C, "penalty": "l2"}

print(f"\nBest Params: {best_params} | Val F1={best_f1:.3f}")

# Retrain best model on full training + validation set
X_train_combined = np.concatenate([X_train_full, X_val_num])
y_train_combined = np.concatenate([y_train_full, y_val])
final_model = LogisticRegression(**best_params, solver="lbfgs", max_iter=1000)
final_model.fit(X_train_combined, y_train_combined)

# Evaluate on test set
y_pred_test = final_model.predict(X_test_num)

print("\n=== Tuned Logistic Regression Baseline ===")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_test, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_test, zero_division=0):.3f}")
print(f"F1: {f1_score(y_test, y_pred_test, zero_division=0):.3f}")

# Convert EHR vector to descriptive text for LLM
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

X_test_texts = [vector_to_text(x) for x in X_test_num]

# Flan-T5 LLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def llm_predict(text_list, batch_size=16, show_progress=False):
    preds = []
    iterator = range(0, len(text_list), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Evaluating Flan-T5")
    for i in iterator:
        batch_texts = text_list[i:i+batch_size]
        inputs = tokenizer(
            [f"Classify isDead (0 or 1) for the following patient:\n{text}" for text in batch_texts],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        outputs = model.generate(**inputs, max_new_tokens=5)
        for output in outputs:
            pred_text = tokenizer.decode(output, skip_special_tokens=True).strip()
            pred_label = 1 if pred_text.startswith("1") else 0
            preds.append(pred_label)
    return np.array(preds)

y_pred_llm = llm_predict(X_test_texts, show_progress=False)

print("\n=== Flan-T5 LLM ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_llm):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_llm, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_llm, zero_division=0):.3f}")
print(f"F1: {f1_score(y_test, y_pred_llm, zero_division=0):.3f}")

# Comparison Summary
print("\n=== Comparison Summary ===")
print(f"Baseline F1: {f1_score(y_test, y_pred_test, zero_division=0):.3f}")
print(f"LLM F1: {f1_score(y_test, y_pred_llm, zero_division=0):.3f}")

# Qualitative Error Analysis
print("\n=== Qualitative Error Analysis ===")
errors = []
for i in range(len(y_test)):
    if y_pred_test[i] != y_test[i] or y_pred_llm[i] != y_test[i]:
        errors.append({
            "true_label": int(y_test[i]),
            "baseline_pred": int(y_pred_test[i]),
            "llm_pred": int(y_pred_llm[i]),
            "text_sample": X_test_texts[i][:200] + "..."
        })

print(f"Total error cases: {len(errors)}")
for e in errors[:5]:
    print("\nTrue:", e["true_label"], "| Baseline:", e["baseline_pred"], "| LLM:", e["llm_pred"])
    print("Sample:", e["text_sample"])
