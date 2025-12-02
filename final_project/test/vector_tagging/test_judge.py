import json
import numpy as np
import pytest

import final_project.vector_tagging.JudgeLLM.label_judge as ej 


# -----------------------
# Common test utilities
# -----------------------

class DummyTokenizer:
    def __init__(self):
        self.eos_token = "</s>"
        self.pad_token = None
        self.padding_side = None


class DummyParam:
    # just enough to satisfy `.device`
    def __init__(self, device="cpu"):
        self.device = device


class DummyModel:
    def __init__(self):
        pass

    def parameters(self):
        # yield one dummy param so next(self.model.parameters()).device works
        yield DummyParam()


def mock_tokenizer(monkeypatch):
    """Patch AutoTokenizer.from_pretrained to return a lightweight dummy."""
    monkeypatch.setattr(
        ej,
        "AutoTokenizer",
        type(
            "AT",
            (),
            {"from_pretrained": staticmethod(lambda *a, **k: DummyTokenizer())},
        ),
    )


def mock_hf_for_load_model(monkeypatch):
    """Patch tokenizer, model, and pipeline for load_model tests."""
    mock_tokenizer(monkeypatch)

    # Mock AutoModelForCausalLM.from_pretrained
    monkeypatch.setattr(
        ej,
        "AutoModelForCausalLM",
        type(
            "AM",
            (),
            {"from_pretrained": staticmethod(lambda *a, **k: DummyModel())},
        ),
    )

    # Mock pipeline to return a simple callable
    def fake_pipeline(*args, **kwargs):
        def _inner(prompt, max_new_tokens=256, return_full_text=False, **_):
            # Return a fixed valid JSON payload
            obj = {
                "scores": {
                    "clinical_correctness": 5,
                    "completeness": 4,
                    "conciseness": 3,
                    "medical_clarity": 2,
                    "formatting_quality": 1,
                },
                "explanation": "ok",
            }
            return [{"generated_text": json.dumps(obj)}]

        return _inner

    monkeypatch.setattr(ej, "pipeline", fake_pipeline)


# -----------------------
# Constructor / init
# -----------------------

# Need to test if the initializer function works properly and do a sanity check on model hyperparameters.
def test_init_tokenizer_only(monkeypatch):
    mock_tokenizer(monkeypatch)

    vec_defs = np.array(["feat0", "feat1", "feat2"])
    judge = ej.EHRLabelJudge(vec_defs, model_id="some-model-id")

    assert judge.model_id == "some-model-id"
    assert (judge.vector_definitions == vec_defs).all()
    assert isinstance(judge.tokenizer, DummyTokenizer)
    assert judge.tokenizer.pad_token == judge.tokenizer.eos_token
    assert judge.tokenizer.padding_side == "left"
    assert judge.model is None
    assert judge.pipe is None


# -----------------------
# _format_vector
# -----------------------

# Basic tests for the formatting the vectors. Important for the LLM understanding the prompt
def test_format_vector_non_zero_only(monkeypatch):
    mock_tokenizer(monkeypatch)

    vec_defs = np.array(["gender", "age", "bp"])
    judge = ej.EHRLabelJudge(vec_defs)

    vec = np.array([0.0, 30.0, 0.5])
    txt = judge._format_vector(vec)

    # Should contain only the non-zero features (age, bp), not gender
    assert "- age: 30.0" in txt
    assert "- bp: 0.5" in txt
    assert "gender" not in txt

#Test chosen to evaluate basic functionality when vector is all zeroes
def test_format_vector_all_zeros(monkeypatch):
    mock_tokenizer(monkeypatch)

    vec_defs = np.array(["a", "b", "c"])
    judge = ej.EHRLabelJudge(vec_defs)

    vec = np.array([0.0, 0.0, 0.0])
    txt = judge._format_vector(vec)

    assert txt == "(All zeros)"

#Test case chosen to ensure that in case dimension mismatch, the function fails gracefully.
def test_format_vector_throws_error_on_dim_mismatch(monkeypatch):
     with pytest.raises(ValueError):
        mock_tokenizer(monkeypatch)
        vec_defs = np.array(["a", "b", "c"])
        judge = ej.EHRLabelJudge(vec_defs)

        vec = np.array([0.0, 0.0])
        txt = judge._format_vector(vec)


# -----------------------
# _build_prompt
# -----------------------
# Basic sanity check for prompt builder
def test_build_prompt_contains_vector_and_label(monkeypatch):
    mock_tokenizer(monkeypatch)

    vec_defs = np.array(["age", "bp"])
    judge = ej.EHRLabelJudge(vec_defs)

    vec_txt = "- age: 40.0\n- bp: 120.0"
    label = "40-year-old with hypertension."

    prompt = judge._build_prompt(vec_txt, label)

    # Contains the vector text
    assert "EHR VECTOR" in prompt
    assert vec_txt in prompt

    # Contains the label inside triple quotes
    assert '"""40-year-old with hypertension."""' in prompt

    # Contains rubric field names
    for key in [
        "clinical_correctness",
        "completeness",
        "conciseness",
        "medical_clarity",
        "formatting_quality",
    ]:
        assert key in prompt


# -----------------------
# _parse_output
# -----------------------

# Tests the JSON output parser with a standard output, chosen as a sanity check to ensure JSONs can be parsed properly.
def test_parse_output_valid_json(monkeypatch):
    mock_tokenizer(monkeypatch)

    vec_defs = np.array(["feat0"])
    judge = ej.EHRLabelJudge(vec_defs)

    obj = {
        "scores": {
            "clinical_correctness": 5,
            "completeness": 4,
            "conciseness": 3,
            "medical_clarity": 2,
            "formatting_quality": 1,
        },
        "explanation": "good label",
    }

    text = json.dumps(obj)
    scores, expl = judge._parse_output(text)

    assert scores == [5, 4, 3, 2, 1]
    assert expl == "good label"

# Tests the JSON output parser with an incomplete output, test case chosen because we want to ensure the pipeline doesn't fail on minor output problems
def test_parse_output_missing_fields(monkeypatch):
    mock_tokenizer(monkeypatch)

    vec_defs = np.array(["feat0"])
    judge = ej.EHRLabelJudge(vec_defs)

    # Only provide two fields, others should default to 0
    obj = {
        "scores": {
            "clinical_correctness": 3,
            "conciseness": 2,
        },
        "explanation": "partial",
    }

    text = json.dumps(obj)
    scores, expl = judge._parse_output(text)

    # Order: clinical_correctness, completeness, conciseness, medical_clarity, formatting_quality
    assert scores == [3, 0, 2, 0, 0]
    assert expl == "partial"

# Test chosen to ensure that we can extract clean outputs when the LLM includes extra tests
def test_parse_output_with_wrapped_text(monkeypatch):
    mock_tokenizer(monkeypatch)

    vec_defs = np.array(["feat0"])
    judge = ej.EHRLabelJudge(vec_defs)

    obj = {
        "scores": {
            "clinical_correctness": 1,
            "completeness": 2,
            "conciseness": 3,
            "medical_clarity": 4,
            "formatting_quality": 5,
        },
        "explanation": "wrapped",
    }

    json_block = json.dumps(obj, indent=2)
    wrapped = f"Some preface...\n```json\n{json_block}\n```\nMore text..."

    scores, expl = judge._parse_output(wrapped)

    assert scores == [1, 2, 3, 4, 5]
    assert expl == "wrapped"

# Test chosen to ensure that invalid outputs fail gracefully.
def test_parse_output_invalid_returns_zeros(monkeypatch):
    mock_tokenizer(monkeypatch)

    vec_defs = np.array(["feat0"])
    judge = ej.EHRLabelJudge(vec_defs)

    text = "this is not json at all"
    scores, expl = judge._parse_output(text)

    assert scores == [0, 0, 0, 0, 0]
    assert "Parse error" in expl


# -----------------------
# judge (single)
# -----------------------

# Test to see that the judge pipeline parses it's outputs correctly, important for to test non-model code inside the inference pipeline
def test_judge_uses_pipe_and_parses(monkeypatch):
    mock_tokenizer(monkeypatch)

    vec_defs = np.array(["feat0", "feat1"])
    judge = ej.EHRLabelJudge(vec_defs)

    # Fake JSON output from the model
    obj = {
        "scores": {
            "clinical_correctness": 5,
            "completeness": 5,
            "conciseness": 4,
            "medical_clarity": 3,
            "formatting_quality": 2,
        },
        "explanation": "looks fine",
    }
    json_text = json.dumps(obj)

    def fake_pipe(prompt, max_new_tokens=256, return_full_text=False, **_):
        # ensure prompt contains vector and label markers
        assert "EHR VECTOR" in prompt
        assert "GENERATED LABEL" in prompt
        return [{"generated_text": json_text}]

    # Inject fake pipe directly, no need to call load_model
    judge.pipe = fake_pipe

    vector = np.array([0.0, 1.0])
    label = "some label"

    scores, expl = judge.judge(vector, label)

    assert scores == [5, 5, 4, 3, 2]
    assert expl == "looks fine"


# -----------------------
# judge_batch
# -----------------------

# Test to see that the judge pipeline parses it's outputs correctly on multiple inputs, important for to test non-model code inside the inference pipeline
def test_judge_batch_multiple(monkeypatch):
    mock_tokenizer(monkeypatch)

    vec_defs = np.array(["feat0", "feat1"])
    judge = ej.EHRLabelJudge(vec_defs)

    # Build two different responses to verify indexing
    objs = [
        {
            "scores": {
                "clinical_correctness": 1,
                "completeness": 2,
                "conciseness": 3,
                "medical_clarity": 4,
                "formatting_quality": 5,
            },
            "explanation": "first",
        },
        {
            "scores": {
                "clinical_correctness": 5,
                "completeness": 4,
                "conciseness": 3,
                "medical_clarity": 2,
                "formatting_quality": 1,
            },
            "explanation": "second",
        },
    ]
    json_texts = [json.dumps(o) for o in objs]

    def fake_pipe(prompts, max_new_tokens=256, return_full_text=False, **_):
        # We expect a list of prompts of same length as vectors/labels
        assert isinstance(prompts, list)
        assert len(prompts) == 2
        return [{"generated_text": t} for t in json_texts]

    judge.pipe = fake_pipe

    vectors = [np.array([1.0, 0.0]), np.array([0.0, 2.0])]
    labels = ["lbl1", "lbl2"]

    results = judge.judge_batch(vectors, labels)

    assert len(results) == 2

    (scores1, expl1), (scores2, expl2) = results

    assert scores1 == [1, 2, 3, 4, 5]
    assert expl1 == "first"
    assert scores2 == [5, 4, 3, 2, 1]
    assert expl2 == "second"

# Sanity check for the EHR label constructor.
def test_load_model_sets_model_and_pipe(monkeypatch):
    """
    We don't actually load a real 70B model.
    Instead we mock HF model + pipeline and check
    that load_model wires self.model and self.pipe.
    """
    mock_hf_for_load_model(monkeypatch)

    vec_defs = np.array(["feat0"])
    judge = ej.EHRLabelJudge(vec_defs, model_id="fake-model-id")

    assert judge.model is None
    assert judge.pipe is None

    judge.load_model()

    assert isinstance(judge.model, DummyModel)
    assert callable(judge.pipe)
