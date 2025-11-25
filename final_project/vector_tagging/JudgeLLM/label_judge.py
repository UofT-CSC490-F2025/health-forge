from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
import json
import re


class EHRLabelJudge:
    """
    Judge model for evaluating the quality of LLM-generated patient labels
    using a rubric. It receives:
        • vector_definitions  – array of feature names
        • vector              – numeric feature vector
        • label               – generated label string to evaluate

    The judge returns:
        • scores_list = [int, int, ...]
        • explanation = "string"
    """

    model = None
    pipe = None
    vector_definitions: np.ndarray = None

    def __init__(self, vector_definitions: np.ndarray, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Initialize judge model & pipeline.
        """
        import torch

        print("----------------------------------------------------")
        print("Loading Judge Model")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print("----------------------------------------------------")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=tokenizer
        )

        print("Judge model device:", next(self.model.parameters()).device)
        print("----------------------------------------------------\n")

        self.vector_definitions = vector_definitions


    # ------------------------------------------------------------
    # Format EHR vector into human-readable text
    # ------------------------------------------------------------
    def _format_vector(self, vector: np.ndarray):
        """
        Produces a readable text block of only non-zero features.
        """
        mapping = ""
        for i, value in enumerate(vector):
            if float(value) != 0.0:
                mapping += f"- {self.vector_definitions[i]}: {float(value)}\n"
        
        return mapping if mapping else "(All values zero)"


    # ------------------------------------------------------------
    # High-quality rubric for the judge to follow
    # ------------------------------------------------------------
    def _build_prompt(self, vector_text: str, label: str):
        return f"""
You are an expert clinical evaluator.

You will rate the quality of a generated EHR label for a respective EHR vector based on the rubric below.

You MUST answer by filling in this exact template:

SCORES: [<int>, <int>, <int>, <int>, <int>]
EXPLANATION: <your justification of high and low scores here>

Where:
- SCORES is a list of integer scores from 1 to 5
- No JSON, no backticks, no extra text before or after the template
- Do NOT repeat the instructions
- Do NOT output anything except the template above

Rubric Categories:
A) Clinical correctness: does the label reflect the EHR data?
B) Completeness: does it mention necessary fields (age, gender, ICD clues)?
C) Conciseness: is it short and correctly summarized?
D) Medical clarity: medically valid, no hallucinations
E) Formatting quality: grammar, clarity

--------------------------------------
EHR VECTOR for reference:
{vector_text}

GENERATED LABEL which you are judging:
"{label}"
--------------------------------------

Fill in the template now:

        """



    # ------------------------------------------------------------
    # Extract JSON safely
    # ------------------------------------------------------------
    def _parse_judge_output(self, text):
        scores_match = re.search(r"SCORES:\s*\[(.*?)\]", text)
        if scores_match:
            nums = scores_match.group(1)
            scores = [int(x.strip()) for x in nums.split(",")]
        else:
            scores = [0,0,0,0,0]

        explanation_match = re.search(r"EXPLANATION:\s*(.*)", text, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation."

        return scores, explanation



    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def judge(self, vector: np.ndarray, label: str):
        """
        Evaluate a generated label and return:
            (scores_list, explanation)
        """
        vector_text = self._format_vector(vector)
        prompt = self._build_prompt(vector_text, label)

        out = self.pipe(
            prompt,
            max_new_tokens=256,
            return_full_text=False
        )

        raw_output = out[0]["generated_text"]
        print("\n--- Raw Judge Output ---\n", raw_output)

        scores, explanation = self._parse_judge_output(raw_output)

        print("\n--- Parsed Judge Result ---")
        print("Scores:", scores)
        print("Explanation:", explanation)

        return scores, explanation



# Demo usage
if __name__ == "__main__":
    vec_defs = np.array(["age", "gender", "ICD10:E11", "ICD10:I20"])
    judge = EHRLabelJudge(vec_defs)

    # ------------------------------------------
    # Case 1 — CORRECT LABEL
    # ------------------------------------------
    vector1 = np.array([69, 1, 1, 0])   # 69-year-old male with E11 (T2D)
    label1 = "69-year-old male with type 2 diabetes."

    print("\n=========== CASE 1: CORRECT LABEL =============")
    s1, e1 = judge.judge(vector1, label1)
    print("Scores:", s1)
    print("Explanation:", e1)


    # ------------------------------------------
    # Case 2 — SLIGHTLY WRONG LABEL
    # Missing ICD information, vague phrasing
    # ------------------------------------------
    vector2 = np.array([69, 1, 1, 0])
    label2 = "An older male patient."

    print("\n=========== CASE 2: SLIGHTLY WRONG LABEL =============")
    s2, e2 = judge.judge(vector2, label2)
    print("Scores:", s2)
    print("Explanation:", e2)


    # ------------------------------------------
    # Case 3 — COMPLETELY WRONG LABEL
    # Wrong age, wrong gender, wrong condition
    # ------------------------------------------
    vector3 = np.array([69, 1, 1, 0])
    label3 = "A 25-year-old female with asthma."

    print("\n=========== CASE 3: INCORRECT LABEL =============")
    s3, e3 = judge.judge(vector3, label3)
    print("Scores:", s3)
    print("Explanation:", e3)

    print("\n============= FINAL SUMMARY =============")
    print("Correct label scores     :", s1)
    print("Slightly wrong scores    :", s2)
    print("Completely wrong scores  :", s3)
