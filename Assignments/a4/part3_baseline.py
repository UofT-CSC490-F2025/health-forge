import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import torch

# Load Real and Synthetic Datasets
real = np.load("ehr_real.npy")
synth = np.load("ehr_synth.npy")

print(f"Real data shape: {real.shape}")
print(f"Synth data shape: {synth.shape}")

# We’ll assume same column structure; drop ID/isDead etc.
X_real = real[:, 1:]
X_synth = synth[:, 1:]

# Compute baseline metric scores
def compute_baseline_scores(X_real, X_synth, a=0.3):

    # Fit mean and covariance of real data
    mu = np.mean(X_real, axis=0)
    cov = np.cov(X_real, rowvar=False)
    cov_inv = np.linalg.pinv(cov)  # pseudoinverse for numerical stability

    # Mahalanobis distance to real data distribution
    maha_dist = np.array([mahalanobis(x, mu, cov_inv) for x in X_synth])

    # Nearest real sample distance (Euclidean)
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_real)
    dist_closest, _ = nn.kneighbors(X_synth)
    dist_closest = dist_closest[:, 0]

    # Combine distances
    maha_dist = (maha_dist - maha_dist.min()) / (maha_dist.max() - maha_dist.min())
    dist_closest = (dist_closest - dist_closest.min()) / (dist_closest.max() - dist_closest.min())
    S = (1 - a) * maha_dist - a * dist_closest
    S = (S - S.min()) / (S.max() - S.min())

    return S

baseline_scores = compute_baseline_scores(X_real, X_synth)
baseline_pred = (baseline_scores < 0.5).astype(int)  

# Convert synthetic EHR to text for LLM evaluation
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

# Sample first 100 synthetic records for LLM evaluation
X_synth_texts = [vector_to_text(x) for x in X_synth[:100]]  

# LLM-based evaluation (Flan-T5)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def llm_rate_samples(text_list, batch_size=8, show_progress=False):
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
            # Extract first number in output
            digits = [int(s) for s in pred_text if s.isdigit()]
            score = digits[0] if digits else 5
            scores.append(score)
    return np.array(scores)

llm_scores = llm_rate_samples(X_synth_texts, show_progress=True)
llm_pred = (llm_scores > 5).astype(int)

# Compare Baseline vs LLM
# For simplicity, assume LLM labels are "ground truth"
y_true = llm_pred  # what LLM thinks is good
y_pred_baseline = baseline_pred[:len(y_true)]

print("\n=== Baseline Quantitative Metric vs LLM Evaluation ===")
print(f"Accuracy: {accuracy_score(y_true, y_pred_baseline):.3f}")
print(f"Precision: {precision_score(y_true, y_pred_baseline, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_true, y_pred_baseline, zero_division=0):.3f}")
print(f"F1: {f1_score(y_true, y_pred_baseline, zero_division=0):.3f}")

# Qualitative Error Analysis
print("\n=== Qualitative Error Analysis ===")
errors = []
for i in range(len(y_true)):
    if y_true[i] != y_pred_baseline[i]:
        errors.append({
            "true_label": int(y_true[i]),
            "baseline_pred": int(y_pred_baseline[i]),
            "llm_score": int(llm_scores[i]),
            "text_sample": X_synth_texts[i][:200] + "..."
        })

print(f"Total error cases: {len(errors)}")
for e in errors[:5]:
    print(f"\nTrue: {e['true_label']} | Baseline: {e['baseline_pred']} | LLM Score: {e['llm_score']}")
    print("Sample:", e["text_sample"])
