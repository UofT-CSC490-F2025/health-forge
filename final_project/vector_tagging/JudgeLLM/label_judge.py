from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import json
import re

class EHRLabelJudge:
    """
    Judge model for evaluating LLM-generated labels using a strict JSON rubric.
    Designed to support very large medical LLMs such as Med42-70B.
    """

    def __init__(self, vector_definitions: np.ndarray, model_id="m42-health/med42-70b"):
        """
        Initialize tokenizer, but delay model loading until load_model() is called.
        This keeps the class usable locally without loading 70B params.
        """

        self.model_id = model_id
        self.vector_definitions = vector_definitions

        print("Initializing tokenizer only…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = None
        self.pipe = None
        print("Tokenizer loaded. Call load_model() inside a Modal GPU function.")


    # ------------------------------------------------------------
    # Load model inside Modal GPU container
    # ------------------------------------------------------------
    def load_model(self):
        """Call inside a Modal GPU worker. Loads Med42-70B into memory."""
        import torch

        print("----------------------------------------------------")
        print(f"Loading LLM: {self.model_id}")
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Device:", torch.cuda.get_device_name(0))
        print("----------------------------------------------------")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=0.1,
            max_new_tokens=256,
        )

        print("Model successfully loaded on device:", next(self.model.parameters()).device)


    # ------------------------------------------------------------
    # Format vector into readable text
    # ------------------------------------------------------------
    def _format_vector(self, vector: np.ndarray):
        mapping = ""
        for i, value in enumerate(vector):
            if float(value) != 0.0:
                mapping += f"- {self.vector_definitions[i]}: {float(value)}\n"
        return mapping or "(All zeros)"


    # ------------------------------------------------------------
    # JSON Rubric Prompt
    # ------------------------------------------------------------
    def _build_prompt(self, vector_text: str, label: str):

        return f"""
You are an expert clinical evaluator.

You will evaluate how well a GENERATED LABEL represents the given EHR VECTOR.

Return ONLY valid JSON in this format:

{{
  "scores": {{
      "clinical_correctness": 1-5,
      "completeness": 1-5,
      "conciseness": 1-5,
      "medical_clarity": 1-5,
      "formatting_quality": 1-5
  }},
  "explanation": "Short explanation of strengths and weaknesses."
}}

Rules:
- All values must be integers 1–5.
- Output must be valid JSON with no text before or after.

---------------------------------------------------------
EHR VECTOR (non-zero features):
{vector_text}

GENERATED LABEL:
\"\"\"{label}\"\"\"
---------------------------------------------------------
Return ONLY the JSON.
"""


    # ------------------------------------------------------------
    # JSON Parser
    # ------------------------------------------------------------
    def _parse_output(self, text: str):

        try:
            # Extract JSON substring
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("No JSON found.")

            json_text = match.group(0)
            data = json.loads(json_text)

            # Validate structure
            scores_dict = data.get("scores", {})
            explanation = data.get("explanation", "")

            # Convert dict → ordered list for ML
            order = [
                "clinical_correctness",
                "completeness",
                "conciseness",
                "medical_clarity",
                "formatting_quality"
            ]

            scores_list = [int(scores_dict.get(k, 0)) for k in order]

            return scores_list, explanation

        except Exception as e:
            print("JSON parse error:", e)
            return [0, 0, 0, 0, 0], "Parse error."


    # ------------------------------------------------------------
    # Judge a single vector-label pair
    # ------------------------------------------------------------
    def judge(self, vector: np.ndarray, label: str):

        vector_text = self._format_vector(vector)
        prompt = self._build_prompt(vector_text, label)

        output = self.pipe(
            prompt,
            max_new_tokens=256,
            return_full_text=False,
        )

        raw = output[0]["generated_text"]
        print("\n--- Raw Output ---\n", raw)

        return self._parse_output(raw)


    # ------------------------------------------------------------
    # Batch Judging
    # ------------------------------------------------------------
    def judge_batch(self, vectors: list, labels: list):
        """
        vectors: list[np.ndarray]
        labels:  list[str]
        Returns list of (scores, explanation)
        """

        prompts = []
        for v, lbl in zip(vectors, labels):
            vec_text = self._format_vector(v)
            prompts.append(self._build_prompt(vec_text, lbl))

        # Batch generation
        outputs = self.pipe(
            prompts,
            max_new_tokens=256,
            return_full_text=False,
        )

        results = []
        for out in outputs:
            raw = out["generated_text"]
            results.append(self._parse_output(raw))

        return results
