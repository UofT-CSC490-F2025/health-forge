import os
import json
import boto3
import numpy as np
import modal
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
BUCKET = "healthforge-final-bucket"
VECTORS_KEY = "merged_patient_vectors.npy"
DEF_CSV_KEY = "patient_vector_columns.csv"
LABELS_KEY = "generated_labels.json"

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# -----------------------------------------------------
# Modal Setup
# -----------------------------------------------------
app = modal.App("ehr-label-judge")

image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "pandas",
        "boto3"
    ])
)

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("aws-secret")]
)
def run_judge_on_s3(
    test_mode: bool = True   # ★ NEW FLAG ★
):
    """
    If test_mode=True → run 10 handcrafted examples first.
    Otherwise → run on S3 vectors+labels.
    """

    import torch
    global np

    # ---------------------------------------------------------
    # Load Mixtral-8x22B-Instruct
    # ---------------------------------------------------------
    print(f"=== Loading {MODEL_ID} Judge Model ===")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=512
    )


    # ---------------------------------------------------------
    # Helper: JSON prompt builder
    # ---------------------------------------------------------
    def build_prompt(vector_text: str, label: str):
        return f"""
You are an expert clinical evaluator judging accuracy of labels assigned to patient EHRs.
Your ENTIRE output MUST be ONLY a valid JSON object. No explanations or notes before or after. Make sure the JSON strictly follows this format and naming conventions:

{{
  "scores": {{
       "clinical_correctness": <int>,
       "completeness_of_key_patient_information": <int>,
       "precision_of_patient_information": <int>,
       "conciseness": <int>,
       "formatting": <int>
  }},
  "explanation": "<string>"
}}

Each score must be an INTEGER from 1 to 5.

--------------------------------------------------------------
EHR VECTOR (non-zero entries):
{vector_text}

LABEL YOU ARE EVALUATING:
\"\"\"{label}\"\"\" 
--------------------------------------------------------------
"""

    def parse_json(output: str):
        # Strip markdown fences if present
        output = output.replace("```json", "").replace("```", "")

        # Attempt direct parse
        try:
            return json.loads(output.strip())
        except:
            pass

        # Extract the FIRST valid { } block only
        import re
        candidates = re.findall(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", output, flags=re.DOTALL)

        for c in candidates:
            try:
                return json.loads(c)
            except:
                continue

        # If everything fails → return fallback
        return {
            "scores": {
                "clinical_correctness": 0,
                "completeness": 0,
                "precision": 0,
                "conciseness": 0,
                "formatting": 0,
            },
            "justification": "JSON parse failed."
        }


    # ==================================================================
    # ★ TEST MODE: 10 handcrafted examples (NO S3 NEEDED)
    # ==================================================================
    if test_mode:
        print("=== RUNNING 10 HANDCRAFTED TEST EXAMPLES ===")

        # ----------------------------------------------------
        # Simulated realistic vector definitions (20 columns)
        # ----------------------------------------------------
        vector_defs = np.array([
            # Demographics
            "isMale",
            "age",
            "isDead",
            "total_admissions",

            # Marital status
            "married",
            "single",
            "divorced",
            "widowed",

            # Race / Ethnicity
            "white",
            "black_african_american",
            "asian_chinese",
            "hispanic_latino_mexican",

            # Infectious diseases
            "bacterial_infections",
            "viral_infection",

            # Chronic diseases
            "diabetes_without_complication",
            "hypertension",
            "heart_failure",
            "asthma",

            # Cancer categories
            "breast_cancer",
            "prostate_cancer"
        ])

        # ----------------------------------------------------
        # Helper to format vectors for prompting
        # ----------------------------------------------------
        def format_vector(vec):
            txt = ""

            # Gender is ALWAYS included, even if 0
            gender_value = int(vec[0])
            gender_str = "male" if gender_value == 1 else "female"
            txt += f"- gender: {gender_str}\n"

            # All other values skip zeros
            for i, v in enumerate(vec[1:], start=1):
                if float(v) != 0:
                    txt += f"- {vector_defs[i]}: {float(v)}\n"

            return txt.strip()


        # ----------------------------------------------------
        # ★ 10 Test Patients (5 correct, 5 misleading)
        # ----------------------------------------------------
        test_vectors = [
            # --- Correct Labels ---
            np.array([1, 67, 0, 4, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
            np.array([0, 45, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]),
            np.array([1, 82, 1, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]),
            np.array([0, 33, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
            np.array([1, 29, 0, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),

            # --- Misleading Labels ---
            np.array([1, 70, 0, 5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]),
            np.array([0, 52, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
            np.array([1, 60, 1, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]),
            np.array([0, 25, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]),
            np.array([1, 90, 1, 10, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]),
        ]

        test_labels = [
            # Correct
            "67-year-old male with diabetes and hypertension.",
            "45-year-old woman with bacterial infection, hypertension, and asthma.",
            "82-year-old man with viral infection, hypertension, and heart failure.",
            "33-year-old woman of Asian and Hispanic background with diabetes and asthma.",
            "29-year-old male, white, with asthma.",

            # Misleading
            "Teenage boy with no medical conditions.",  # wrong age, wrong sex
            "Young Hispanic male with cancer.",          # wrong race, wrong disease
            "Healthy 60-year-old woman.",                # wrong sex, wrong conditions
            "25-year-old male with diabetes.",           # wrong sex, wrong diseases
            "90-year-old cancer-free female patient.",   # wrong sex, wrong cancer
        ]

        # ----------------------------------------------------
        # Run through the judge model
        # ----------------------------------------------------
        results = {}

        # Track scores for mean calculations
        correct_scores = []
        misleading_scores = []

        def extract_scores(obj):
            """Return scores in fixed order or None on failure."""
            try:
                s = obj["scores"]
                return [
                    s.get("clinical_correctness", 0),
                    s.get("completeness_of_key_patient_information", 0),
                    s.get("precision_of_patient_information", 0),
                    s.get("conciseness", 0),
                    s.get("formatting", 0),
                ]
            except:
                return None

        for i in range(10):
            vec = test_vectors[i]
            label = test_labels[i]

            print(f"\n==============================")
            print(f"=== TEST SAMPLE {i} ===")
            print("==============================")

            vectxt = format_vector(vec)
            prompt = build_prompt(vectxt, label)

            # --- Generate raw text ---
            raw = pipe(
                prompt,
                max_new_tokens=256,
                return_full_text=False,
                temperature=0.2
            )[0]["generated_text"]

            print("\n--- RAW MODEL OUTPUT ---")
            print(raw)
            print("-------------------------\n")

            # --- Parse JSON ---
            parsed = parse_json(raw)
            results[i] = parsed

            # Collect scores for averages
            sc = extract_scores(parsed)
            if sc:
                if i < 5:
                    correct_scores.append(sc)
                else:
                    misleading_scores.append(sc)

        # ----------------------------------------------------
        # Compute per-category averages
        # ----------------------------------------------------
        import numpy as np

        def mean_scores(arr):
            arr = np.array(arr)
            return arr.mean(axis=0).round(3).tolist()

        correct_avg = mean_scores(correct_scores) if correct_scores else None
        misleading_avg = mean_scores(misleading_scores) if misleading_scores else None

        print("\n==============================")
        print("MEAN SCORES (FIRST 5 = CORRECT LABELS)")
        print("==============================")
        print({
            "clinical_correctness": correct_avg[0],
            "completeness_of_key_patient_information": correct_avg[1],
            "precision_of_patient_information": correct_avg[2],
            "conciseness": correct_avg[3],
            "formatting": correct_avg[4],
        })

        print("\n==============================")
        print("MEAN SCORES (LAST 5 = MISLEADING LABELS)")
        print("==============================")
        print({
            "clinical_correctness": misleading_avg[0],
            "completeness_of_key_patient_information": misleading_avg[1],
            "precision_of_patient_information": misleading_avg[2],
            "conciseness": misleading_avg[3],
            "formatting": misleading_avg[4],
        })

        # Save & upload
        out_path = "/tmp/judge_test_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        boto3.client("s3").upload_file(out_path, BUCKET, "judge_test_results.json")

        return {
            "status": "test_ok",
            "count": len(results),
            "correct_avg": correct_avg,
            "misleading_avg": misleading_avg,
        }


    # ==================================================================
    # ★ NORMAL MODE: Evaluate full S3 dataset
    # ==================================================================
    print("=== TEST MODE OFF → Running full S3 evaluation ===")

    s3 = boto3.client("s3")

    # Load vectors
    with open("/tmp/vectors.npy", "wb") as f:
        s3.download_fileobj(BUCKET, VECTORS_KEY, f)
    vectors = np.load("/tmp/vectors.npy")

    # Load definitions
    with open("/tmp/defs.csv", "wb") as f:
        s3.download_fileobj(BUCKET, DEF_CSV_KEY, f)
    df = pd.read_csv("/tmp/defs.csv")
    vector_definitions = df.columns.values

    # Load labels
    with open("/tmp/labels.json", "wb") as f:
        s3.download_fileobj(BUCKET, LABELS_KEY, f)
    label_dict = json.load(open("/tmp/labels.json"))
    
    def format_vector_full(vec):
        txt = ""

        # Gender is ALWAYS included, even if 0
        gender_value = int(vec[0])
        gender_str = "male" if gender_value == 1 else "female"
        txt += f"- gender: {gender_str}\n"

        # All other values skip zeros
        for i, v in enumerate(vec[1:], start=1):
            if float(v) != 0:
                txt += f"- {vector_defs[i]}: {float(v)}\n"

        return txt.strip()


    results = {}

    for i in range(len(vectors)):
        vec = vectors[i]
        label = label_dict.get(str(i), "(missing)")
        vtxt = format_vector_full(vec)
        prompt = build_prompt(vtxt, label)

        print(f"--- Evaluating vector {i} ---")
        out = pipe(
            prompt,
            max_new_tokens=256,
            return_full_text=False,
            temperature=0.2
        )[0]["generated_text"]

        results[i] = parse_json(out)

    out_path = "/tmp/judge_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    s3.upload_file(out_path, BUCKET, "judge_results.json")
    print("=== Uploaded judge_results.json to S3 ===")

    return {"status": "ok", "count": len(results)}


# ---------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------
@app.local_entrypoint()
def main():
    # first run in test mode
    run_judge_on_s3.remote(test_mode=True)
