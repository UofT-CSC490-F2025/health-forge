import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# ------------------------------
# Load dataset
# ------------------------------
data = np.load("ehr_norm.npy") 
print("Loaded data shape:", data.shape)

# Extract label and features
# Column indices based on your description:
# 0 - Subject ID, 1 - Gender, 2 - Age, 3 - isDead
y = data[:, 3].astype(int)               # isDead column as label
X = np.delete(data[:, 1:], 2, axis=1)  

# ------------------------------
# Train/test split
# ------------------------------
X_train_num, X_test_num, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=49
)

# ------------------------------
# Logistic Regression Baseline
# ------------------------------
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_num, y_train)
y_pred_baseline = baseline_model.predict(X_test_num)

print("=== Logistic Regression Baseline ===")
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Precision:", precision_score(y_test, y_pred_baseline, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_baseline, zero_division=0))
print("F1:", f1_score(y_test, y_pred_baseline, zero_division=0))

# ------------------------------
# Convert EHR vector to detailed text
# ------------------------------
def vector_to_text(vector):
    """
    Convert numeric EHR vector into descriptive text suitable for Flan-T5.
    Shows first 10 entries per table to avoid token overflow.
    """
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
        vec_slice = vector[offset : offset + length]
        text += f"{table_name}: {vec_slice.tolist()[:10]} ...\n"  # show first 10 values
        offset += length

    return text

# Convert test set to text
X_test_texts = [vector_to_text(x) for x in X_test_num]

# ------------------------------
# Flan-T5 LLM Judge
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def llm_predict(text_list, batch_size=32):
    preds = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="Evaluating Flan-T5"):
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

# Run LLM judge on full test set
y_pred_llm = llm_predict(X_test_texts)

print("\n=== Flan-T5 LLM Judge ===")
print("Accuracy:", accuracy_score(y_test, y_pred_llm))
print("Precision:", precision_score(y_test, y_pred_llm, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_llm, zero_division=0))
print("F1:", f1_score(y_test, y_pred_llm, zero_division=0))

print("\n=== Comparison Summary ===")
print("Baseline F1:", f1_score(y_test, y_pred_baseline))
print("LLM Judge F1:", f1_score(y_test, y_pred_llm))
