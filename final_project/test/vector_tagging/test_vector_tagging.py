import numpy as np
import torch
import pytest

import sys
print(sys.path)

# 🔧 Adjust this import to match your file name
# e.g. from biomistral_vector_tagger import BioMistralVectorTagger
from ...vector_tagging.LLM.vector_tagger import BioMistralVectorTagger


# ---------- Helpers / Fakes ----------

def make_vector_definitions(n_features: int):
    """
    Build a vector_definitions array long enough for tests.
    We only care about indices used in format_vector_full/tag_vectors.
    """
    defs = []
    for i in range(n_features):
        if i == 4:
            defs.append("single")
        elif i == 5:
            defs.append("married")
        elif 9 <= i < 42:
            defs.append(f"ethnicity_{i}")
        elif i >= 42:
            defs.append(f"dx_{i}")
        else:
            defs.append(f"feat_{i}")
    return np.array(defs, dtype=object)


class FakeEncoded(dict):
    """Small wrapper so .to(device) works (returns self)."""
    def to(self, device):
        return self


class FakeBertTokenizer:
    """Minimal tokenizer for encode_text test."""
    def __call__(self, text_batch, padding, truncation, max_length, return_tensors):
        batch_size = len(text_batch)
        seq_len = 4
        input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        return FakeEncoded(input_ids=input_ids, attention_mask=attention_mask)


class FakeEncoder:
    """Minimal encoder that returns a predictable last_hidden_state."""
    def __init__(self):
        self.device = "cpu"

    def __call__(self, **encoded):
        batch_size, seq_len = encoded["input_ids"].shape
        hidden_size = 8
        vals = torch.arange(
            batch_size * seq_len * hidden_size,
            dtype=torch.float32
        ).reshape(batch_size, seq_len, hidden_size)

        class Output:
            pass

        out = Output()
        out.last_hidden_state = vals
        return out


class FakeLLMTokenizer:
    """Fake tokenizer for LLM path in tag_vectors."""
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        # Just return the content; we don't care about exact format here.
        return messages[0]["content"]

    def __call__(self, llm_prompts, padding, truncation, max_length, return_tensors):
        batch_size = len(llm_prompts)
        seq_len = 5
        input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, gen_only, skip_special_tokens):
        batch_size = gen_only.shape[0]
        return [f"summary_{i}" for i in range(batch_size)]


class FakeLLMModel:
    """Fake LLM model with .generate and .device attributes."""
    def __init__(self):
        self.device = "cpu"

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
        pad_token_id,
    ):
        batch_size, seq_len = input_ids.shape
        gen_len = 3
        return torch.zeros((batch_size, seq_len + gen_len), dtype=torch.long)


# ---------- Tests for format_vector_full ----------

def test_format_vector_full_no_diagnoses():
    n_features = 50
    vector_defs = make_vector_definitions(n_features)

    # Build vector with:
    # gender = 1 (male), age = 30, alive = 0 ("is alive")
    # marital status index 4 = single
    # ethnicities indices 9 and 10 = 1
    vec = np.zeros(n_features, dtype=float)
    vec[0] = 1.0   # male
    vec[1] = 30.0  # age
    vec[2] = 0.0   # alive flag => "is alive"
    vec[4] = 1.0   # "single"
    vec[9] = 1.0   # "ethnicity_9"
    vec[10] = 1.0  # "ethnicity_10"
    # diagnoses (42:) left as 0.0 => no diagnoses

    bio = BioMistralVectorTagger.__new__(BioMistralVectorTagger)
    bio.vector_definitions = vector_defs

    text = bio.format_vector_full(vec)

    assert "gender: male" in text
    assert "age: 30" in text
    assert "is alive" in text
    assert "single" in text          # marital status
    assert "ethnicity_9" in text
    assert "ethnicity_10" in text
    assert "has not been diagnosed with any conditions" in text


def test_format_vector_full_with_diagnoses():
    n_features = 50
    vector_defs = make_vector_definitions(n_features)

    vec = np.zeros(n_features, dtype=float)
    vec[0] = 0.0   # female
    vec[1] = 65.0
    vec[2] = 1.0   # "has deceased"
    vec[5] = 1.0   # "married"
    vec[9] = 1.0   # ethnicity_9
    # diagnoses
    vec[42] = 1.0
    vec[45] = 1.0

    bio = BioMistralVectorTagger.__new__(BioMistralVectorTagger)
    bio.vector_definitions = vector_defs

    text = bio.format_vector_full(vec)

    assert "gender: female" in text
    assert "age: 65" in text
    assert "has deceased" in text
    assert "married" in text
    assert "- dx_42" in text
    assert "- dx_45" in text
    # Since there *are* diagnoses, this line should NOT appear
    assert "has not been diagnosed with any conditions" not in text


# ---------- Tests for tag_vectors (no-diagnosis path only) ----------

def test_tag_vectors_all_no_diagnoses():
    n_features = 50
    vector_defs = make_vector_definitions(n_features)

    # Two patients, both with no diagnoses (42:) = 0
    v1 = np.zeros(n_features, dtype=float)
    v1[0] = 1.0   # male
    v1[1] = 30.0
    v1[2] = 0.0   # alive
    v1[9] = 1.0
    v1[10] = 1.0

    v2 = np.zeros(n_features, dtype=float)
    v2[0] = 0.0   # female
    v2[1] = 80.0
    v2[2] = 1.0   # deceased
    v2[9] = 1.0

    batch = np.vstack([v1, v2])

    bio = BioMistralVectorTagger.__new__(BioMistralVectorTagger)
    bio.vector_definitions = vector_defs

    # We do NOT need model / llm_tokenizer because the no-dx path skips LLM
    result = bio.tag_vectors(batch)

    # result is list of [vector, summary]
    assert len(result) == 2

    vec0, summary0 = result[0]
    vec1, summary1 = result[1]

    # Vectors should be preserved
    assert np.array_equal(vec0, v1)
    assert np.array_equal(vec1, v2)

    # Check deterministic summaries
    assert "30-year-old" in summary0
    assert "male" in summary0
    assert "no documented diagnoses" in summary0

    assert "80-year-old" in summary1
    assert "female" in summary1
    assert "no documented diagnoses" in summary1


# ---------- Tests for tag_vectors (mixed: some need LLM) ----------

def test_tag_vectors_mixed_diagnoses_uses_llm_outputs():
    n_features = 50
    vector_defs = make_vector_definitions(n_features)

    # v_no_dx: no diagnoses
    v_no_dx = np.zeros(n_features, dtype=float)
    v_no_dx[0] = 1.0
    v_no_dx[1] = 40.0
    v_no_dx[2] = 0.0
    v_no_dx[9] = 1.0

    # v_with_dx: at least one dx
    v_with_dx = np.zeros(n_features, dtype=float)
    v_with_dx[0] = 0.0
    v_with_dx[1] = 55.0
    v_with_dx[2] = 0.0
    v_with_dx[9] = 1.0
    v_with_dx[42] = 1.0  # dx present

    batch = np.vstack([v_no_dx, v_with_dx])

    bio = BioMistralVectorTagger.__new__(BioMistralVectorTagger)
    bio.vector_definitions = vector_defs

    # Attach fake LLM components so tag_vectors can run without real models
    bio.llm_tokenizer = FakeLLMTokenizer()
    bio.model = FakeLLMModel()

    result = bio.tag_vectors(batch)

    assert len(result) == 2

    vec0, summary0 = result[0]
    vec1, summary1 = result[1]

    # First patient: no diagnoses -> deterministic template
    assert "no documented diagnoses" in summary0

    # Second patient: has diagnoses -> should come from fake LLM
    # Our FakeLLMTokenizer.batch_decode returns "summary_0" for the first LLM sample.
    assert summary1 == "summary_0"


# ---------- Tests for encode_text ----------

def test_encode_text_returns_expected_shape():
    bio = BioMistralVectorTagger.__new__(BioMistralVectorTagger)

    # Fake encoder + tokenizer
    bio.encoder = FakeEncoder()
    bio.bert_tokenizer = FakeBertTokenizer()

    texts = ["first sentence", "second sentence"]
    emb = bio.encode_text(texts)

    # Should be numpy array with shape (batch_size, hidden_size)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == len(texts)
    assert emb.shape[1] == 8  # from FakeEncoder hidden_size
