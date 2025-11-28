# test_patient_samples.py

import torch
import numpy as np
import pytest

# change this import to match your actual filename
import final_project.dummy_data as ps


def test_create_dummy_samples_shape_and_value_ranges():
    sample_size = 10
    max_age = 80

    samples = ps.create_dummy_samples(sample_size, max_age=max_age)

    # shape: (sample_size, 8)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (sample_size, 8)

    # All values should be between 0 and 1 (since age is normalized)
    assert torch.all(samples >= 0.0)
    assert torch.all(samples <= 1.0)

    # First feature is male flag -> should be 0 or 1 (integers, but stored as float)
    male_feature = samples[:, 0]
    assert torch.all((male_feature == 0) | (male_feature == 1))

    # Age feature should be in [0, 1], scaled by max_age
    age_feature = samples[:, 1]
    assert torch.all(age_feature >= 0.0)
    assert torch.all(age_feature <= 1.0)


def test_data_descriptions_known_values():
    # Build a tiny, fully controlled sample batch
    # [male, age, married, dead, diabetes, cancer, depression, schizophrenia]
    samples = torch.tensor([
        [1.0, 0.25, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # male, 25, married, alive, diabetes, no cancer, depression, no schiz
        [0.0, 0.50, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # female, 50, not married, dead, no diabetes, cancer, no dep, schiz
    ])

    descs = ps.data_descriptions(samples, max_age=100)

    assert len(descs) == 2

    # Check first sample descriptions
    d0 = descs[0]
    assert "is male" in d0
    assert "is 25 years old" in d0
    assert "is married" in d0
    assert "is not dead" in d0
    assert "has diabetes" in d0
    assert "does not have cancer" in d0
    assert "has depression" in d0
    assert "does not have schizophrenia" in d0

    # Check second sample descriptions
    d1 = descs[1]
    assert "is female" in d1
    assert "is 50 years old" in d1
    assert "is not married" in d1
    assert "is dead" in d1
    assert "does not have diabetes" in d1
    assert "has cancer" in d1
    assert "does not have depression" in d1
    assert "has schizophrenia" in d1


def test_create_samples_and_desc_consistency():
    num_samples = 5
    max_age = 100

    samples, descs = ps.create_samples_and_desc(num_samples, max_age=max_age)

    # Basic shape check
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (num_samples, 8)

    # There should be one description list per sample
    assert isinstance(descs, list)
    assert len(descs) == num_samples

    # Each description should have 8 entries (one per feature)
    for d in descs:
        assert isinstance(d, list)
        assert len(d) == 8


def test_create_llm_descs_with_mocked_pipeline(monkeypatch):
    """
    We don't want tests to actually download or run the HF model,
    so we mock transformers.pipeline and control the output shape.
    """

    captured_args = {}

    def fake_pipeline(task, model):
        # Basic sanity that we set up the pipeline as expected
        assert task == "text-generation"
        assert isinstance(model, str)

        def _inner(messages):
            # Record what messages we got, so we can assert later
            captured_args["messages"] = messages
            results = []
            # Create fake HF-like output:
            # results[i] -> [ { "generated_text": [ <user>, {"content": "..."} ] } ]
            for i, _ in enumerate(messages):
                content = f"synthetic description {i}"
                results.append([
                    {
                        "generated_text": [
                            {"role": "user", "content": "dummy"},
                            {"role": "assistant", "content": content},
                        ]
                    }
                ])
            return results

        return _inner

    # Apply the monkeypatch
    monkeypatch.setattr(ps, "pipeline", fake_pipeline)

    # Input descriptions: just 2 examples
    descs = [
        ["is male", "is 20 years old"],
        ["is female", "is 50 years old"],
    ]

    llm_descs = ps.create_llm_descs(descs)

    # We expect 2 outputs, one per input
    assert llm_descs == ["synthetic description 0", "synthetic description 1"]

    # Check that the messages passed into the pipeline are formatted correctly
    assert "messages" in captured_args
    messages = captured_args["messages"]
    assert len(messages) == 2
    for msg in messages:
        # each msg should be a list of one dict with keys 'role' and 'content'
        assert isinstance(msg, list)
        assert msg[0]["role"] == "user"
        assert "properties:" in msg[0]["content"]


def test_get_text_embeds_with_mocked_sentence_transformer(monkeypatch):
    """
    Mock SentenceTransformer so we don't hit the network or load a real model.
    """

    class DummyST:
        def __init__(self, model_name):
            # sanity: we can assert that the requested model is the one we expect
            assert "all-MiniLM-L6-v2" in model_name

        def encode(self, descs):
            # Return a simple (len(descs), 2) array where each row is [i, len(desc)]
            # just to have deterministic shape and content.
            emb = []
            for i, d in enumerate(descs):
                emb.append([i, len(d)])
            return np.array(emb, dtype=float)

    monkeypatch.setattr(ps, "SentenceTransformer", DummyST)

    descs = ["short", "a bit longer"]
    embeddings = ps.get_text_embeds(descs)

    # Should be a numpy array of shape (len(descs), 2)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 2)

    # Check our dummy encoding logic
    # Index 0 -> [0, len("short")] = [0, 5]
    # Index 1 -> [1, len("a bit longer")] = [1, 12]
    assert np.allclose(embeddings[0], np.array([0.0, 5.0]))
    assert np.allclose(embeddings[1], np.array([1.0, 12.0]))
