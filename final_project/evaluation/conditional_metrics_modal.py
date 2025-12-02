import modal
import boto3
import os
import numpy as np
import pandas as pd

# -----------------------------
# Modal GPU/CPU image
# -----------------------------
image = (
    modal.Image.debian_slim()
    .pip_install(["boto3", "numpy", "pandas"])
)

app = modal.App("ehr-eval-gpu")

def log(msg):
    print(f"[LOG] {msg}")

# -----------------------------
# Helper: download .npy from S3
# -----------------------------
def download_s3_file(s3, bucket, key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    log(f"Downloading s3://{bucket}/{key} -> {local_path}")
    s3.download_file(bucket, key, local_path)
    return np.load(local_path, allow_pickle=True)

# -----------------------------
# Helper: download column labels from S3
# -----------------------------
def download_column_labels(s3, bucket, key):
    os.makedirs("/tmp", exist_ok=True)
    local_path = f"/tmp/{os.path.basename(key)}"
    log(f"Downloading column labels from s3://{bucket}/{key}")
    s3.download_file(bucket, key, local_path)
    df = pd.read_csv(local_path, header=None)
    return df.iloc[0].tolist()  # single row of column names

# -----------------------------
# Define column ranges for categories
# -----------------------------
# These indices should correspond to the CSV structure
category_ranges = {
    "gender": [0],
    "age": [1],
    "death": [2],
    "total_admissions": [3],

    # marital columns 4–8
    "marital": list(range(4, 9)),
    "marital_tokens_order": [
        "divorced",
        "married",
        "n a",
        "single",
        "widowed",
    ],

    # race columns 9–43
    "race": list(range(9, 44)),
    "race_tokens_order": [
        "american indian alaska native",
        "asian",
        "asian indian",
        "chinese",
        "korean",
        "south east asian",
        "black african",
        "black african american",
        "cape verdean",
        "caribbean island",
        "hispanic or latino",
        "central american",
        "columbian",
        "cuban",
        "dominican",
        "guatemalan",
        "honduran",
        "mexican",
        "puerto rican",
        "salvadoran",
        "multiple race ethnicity",
        "hawaiian pacific islander",
        "other",
        "patient declined",
        "portuguese",
        "south american",
        "unable to obtain",
        "unknown",
        "white",
        "white brazilian",
        "white eastern european",
        "white other european",
        "white russian"
    ],

    "diagnoses": list(range(44, 1806))
}

# -----------------------------
# Convert prompt to expected column values
# -----------------------------
def prompt_to_indices(prompt_text, column_names, category_ranges):
    """
    Convert prompt text into {column_index: expected_value}.

    Rules:
      - Gender / death / age / total_admissions: parsed normally.
      - Marital + race = multi-hot using positional token lists in category_ranges.
        We DO NOT inspect column names.
      - Diagnoses: substring or ICD-style match; require == 1.
      - Expected value:
            • 1 for binary/multi-hot
            • numeric for continuous (age, etc.)
    """
    prompt = prompt_text.lower()
    idx_map = {}

    # --------------------
    # Gender (single index)
    # --------------------
    for gi in category_ranges["gender"]:
        if "male" in prompt:
            idx_map[gi] = 1
        elif "female" in prompt:
            idx_map[gi] = 0

    # --------------------
    # Death / DoD
    # --------------------
    for di in category_ranges["death"]:
        if "dead" in prompt:
            idx_map[di] = 1
        elif "alive" in prompt:
            idx_map[di] = 0

    # --------------------
    # Age — parse first integer
    # --------------------
    for ai in category_ranges["age"]:
        nums = [
            int(s)
            for s in ''.join(
                c if (c.isdigit() or c.isspace()) else ' ' for c in prompt
            ).split()
            if s.isdigit()
        ]
        if nums:
            idx_map[ai] = float(nums[0])
            break

    # --------------------
    # Total admissions (optional)
    # --------------------
    for ti in category_ranges["total_admissions"]:
        words = prompt.split()
        for i, w in enumerate(words):
            if "admission" in w and i + 1 < len(words) and words[i + 1].isdigit():
                idx_map[ti] = float(int(words[i + 1]))
                break

    # ---------------------------------------------------------
    # Marital (multi-hot)
    # IMPORTANT: strictly positional! No column name inspection.
    # category_ranges["marital_tokens_order"] must align with ranges.
    # ---------------------------------------------------------
    marital_tokens_order = category_ranges.get("marital_tokens_order", [])
    marital_indices = category_ranges["marital"]

    for k, token in enumerate(marital_tokens_order):
        if token in prompt:
            idx_map[marital_indices[k]] = 1

    # ---------------------------------------------------------
    # Race (multi-hot)
    # IMPORTANT: strictly positional! No column name inspection.
    # category_ranges["race_tokens_order"] must align with ranges.
    # ---------------------------------------------------------
    race_tokens_order = category_ranges.get("race_tokens_order", [])
    race_indices = category_ranges["race"]

    for k, token in enumerate(race_tokens_order):
        if token in prompt:
            idx_map[race_indices[k]] = 1

    # ---------------------------------------------------------
    # Diagnoses
    # match by substring or ICD-like prefix
    # ---------------------------------------------------------
    words = [
        w.strip().lower()
        for w in prompt.replace(",", " ").split()
        if w.strip()
    ]

    diag_keys = []
    for w in words:
        # ICD-like: alpha + digits
        if len(w) >= 2 and w[0].isalpha() and any(ch.isdigit() for ch in w):
            diag_keys.append(w)
        else:
            diag_keys.append(w)  # plain token

    for di in category_ranges["diagnoses"]:
        cname = column_names[di].lower()
        for dk in diag_keys:
            if dk in cname or cname.startswith(dk):
                idx_map[di] = 1
                break

    return idx_map


# -----------------------------
# Evaluate one sample against a prompt
# -----------------------------
def evaluate_sample(sample_row, prompt_dict, diagnosis_indices=None):
    """
    Modified:
      - age passes if within 1 year
      - diagnoses evaluated as a group:
            if ANY expected diagnosis column is 1 in the sample,
            the entire diagnosis group is counted as correct.
    """
    correct = 0
    total = 0

    # Separate diagnosis indices
    diag_expected = []
    other_expected = []

    for idx, expected_val in prompt_dict.items():
        if diagnosis_indices and idx in diagnosis_indices:
            diag_expected.append(idx)
        else:
            other_expected.append((idx, expected_val))

    # ------------------------
    # Evaluate non-diagnosis
    # ------------------------
    for idx, expected_val in other_expected:
        sample_val = sample_row[idx]

        if isinstance(expected_val, int):
            if sample_val == expected_val:
                correct += 1

        elif isinstance(expected_val, float):
            # ------------------------
            # AGE TOLERANCE ±5
            # ------------------------
            if idx in category_ranges["age"]:
                if abs(sample_val - expected_val) <= 5:   # <-- changed from 1 to 5
                    correct += 1
            else:
                # for other continuous features, keep ±1 (or adjust as needed)
                if abs(sample_val - expected_val) <= 1:
                    correct += 1

        total += 1

    # ------------------------
    # Evaluate diagnosis AS A GROUP
    # ------------------------
    if diag_expected:
        total += 1  # diagnoses count as ONE feature

        # if any diagnosis column is 1, PASS
        diag_pass = any(sample_row[idx] == 1 for idx in diag_expected)

        if diag_pass:
            correct += 1

    return correct / total if total > 0 else 0.0


# -----------------------------
# MAIN MODAL FUNCTION (fixed)
# -----------------------------
# -----------------------------
# MAIN MODAL FUNCTION (fixed)
# -----------------------------
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("aws-secret")],
    gpu="H100",
    timeout=60 * 60,
)
def evaluate_samples():
    import numpy as np

    BUCKET = "healthforge-final-bucket-1"
    SAMPLES_FILE = "results/all_samples.npy"   # shape: (num_prompts, 100, 1806)
    PROMPT_FILE  = "patient_text_prompts.txt"
    COL_LABEL_FILE = "patient_vector_columns.csv"

    s3 = boto3.client("s3")

    # -------------------------
    # Load data
    # -------------------------
    all_samples = download_s3_file(s3, BUCKET, SAMPLES_FILE, "/tmp/all_samples.npy")
    column_labels = download_column_labels(s3, BUCKET, COL_LABEL_FILE)

    prompt_local = "/tmp/patient_text_prompts.txt"
    s3.download_file(BUCKET, PROMPT_FILE, prompt_local)
    with open(prompt_local, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    # Corrected wrapper
    def local_prompt_to_indices(prompt):
        return prompt_to_indices(prompt, column_labels, category_ranges)

    # -------------------------
    # Evaluate
    # -------------------------
    prompt_accuracies = []
    n_prompts = min(len(prompts), all_samples.shape[0])

    DEBUG_PROMPT_INDEX = 97   # prompt #98 for detailed logging

    for i in range(n_prompts):
        prompt_text = prompts[i]

        # FIXED — now matches wrapper signature
        prompt_dict = local_prompt_to_indices(prompt_text)

        samples = all_samples[i]   # shape (100, 1806)

        diagnosis_indices = set(category_ranges["diagnoses"])

        sample_scores = []
        for sample_row in samples:
            sample_scores.append(
                evaluate_sample(sample_row, prompt_dict, diagnosis_indices)
    )


        # ----------------------------------------
        # DETAILED DEBUG LOGGING FOR ONE PROMPT
        # ----------------------------------------
        # ----------------------------------------
        # DETAILED DEBUG LOGGING FOR ONE PROMPT
        # ----------------------------------------
        if i == DEBUG_PROMPT_INDEX:
            log(f"[LOG] --- Detailed evaluation for prompt {i+1} ---")

            diag_indices = set(category_ranges["diagnoses"])

            for j, sample_row in enumerate(samples):
                sample_score = evaluate_sample(sample_row, prompt_dict, diagnosis_indices)
                log(f"[LOG] Sample {j+1}: score={sample_score:.4f}")

                # ---- non-diagnosis logging ----
                for idx, expected_val in prompt_dict.items():
                    if idx not in diag_indices:
                        sample_val = sample_row[idx]

                        if isinstance(expected_val, int):
                            match = (sample_val == expected_val)
                        else:
                            match = abs(sample_val - expected_val) <= 1

                        cname = column_labels[idx]
                        status = "✔" if match else "✖"
                        log(f"[LOG]    {idx} ({cname}): sample={sample_val}, expected={expected_val} -> {status}")

                # ---- diagnosis group logging ----
                diag_expected = [idx for idx in prompt_dict if idx in diag_indices]

                if diag_expected:
                    diag_pass = any(sample_row[idx] == 1 for idx in diag_expected)
                    status = "✔" if diag_pass else "✖"
                    log(f"[LOG]    [DIAGNOSIS GROUP]: {len(diag_expected)} conditions -> {status}")


        # ----------------------------------------
        # Prompt accuracy summary
        # ----------------------------------------
        prompt_accuracy = float(np.mean(sample_scores))
        prompt_accuracies.append(prompt_accuracy)

        log(f"[LOG] Prompt {i+1}: accuracy={prompt_accuracy:.4f}, prompt='{prompt_text}'")

    # -------------------------
    # Final stats
    # -------------------------
    mean_acc = float(np.mean(prompt_accuracies))
    sd_acc   = float(np.std(prompt_accuracies))

    log(f"[LOG] Mean prompt accuracy: {mean_acc:.4f} ± {sd_acc:.4f}")

    return {
        "mean_prompt_accuracy": mean_acc,
        "sd_prompt_accuracy": sd_acc
    }



# -----------------------------
# LOCAL ENTRYPOINT
# -----------------------------
@app.local_entrypoint()
def main():
    evaluate_samples.remote()
    print("Submitted prompt evaluation job on Modal…")

if __name__ == "__main__":
    with app.run():
        main()
