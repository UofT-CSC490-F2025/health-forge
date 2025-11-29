 
import torch
import numpy as np
import pytest

import final_project.sample as sm  


# -----------------------
# Dummy classes for mocks
# -----------------------

class DummySentenceTransformer:
    def __init__(self, dim):
        self.dim = dim

    def encode(self, texts):
        # Return a deterministic embedding of shape (len(texts), dim)
        return np.ones((len(texts), self.dim), dtype=np.float32)


class DummyDiffusionModel(torch.nn.Module):
    """
    Minimal stand-in for DiffusionModel used in sampling.

    - Ignores cfg.
    - forward(z_t, text_embed) returns zeros with same shape as z_t.
    - load_state_dict / eval / to are no-ops.
    """
    def __init__(self, cfg):
        super().__init__()
        # keep input_dim if you like, but we don't actually use it
        self.cfg = cfg

    def forward(self, z_t, text_embed):
        # Predict zero noise to keep math simple
        return torch.zeros_like(z_t)

    def load_state_dict(self, state_dict):
        # No-op (we don't care about weights here)
        return self

    def eval(self):
        return self

    def to(self, device=None):
        return self


# -----------------------
# Helper to build a tiny cfg
# -----------------------

def make_cfg(text_embed_dim=16, T=3):
    """
    Small config so sampling loop stays cheap and deterministic.
    """
    return {
        "data_utils": {
            "T": T,
            "lambda_min": -1.0,
            "lambda_max": 1.0,
        },
        "sampler": {
            "guidance_scale": 1.0,
            "var_interpolation_coeff": 0.5,
        },
        "model": {
            # only text_embed_dim matters for the sampler
            "text_embed_dim": text_embed_dim,
            # you may have other keys in the real config; they are ignored by DummyDiffusionModel
        },
    }


# -----------------------
# Tests
# -----------------------

def test_sample_runs_and_returns_tensor(monkeypatch):
    """
    End-to-end-ish test of sample(cfg):
    - Mocks SentenceTransformer to avoid network/model load
    - Mocks DiffusionModel to avoid loading weights / architecture
    - Mocks torch.load to avoid reading from disk
    - Forces CPU path (cuda.is_available=False)
    """

    text_embed_dim = 16
    cfg = make_cfg(text_embed_dim=text_embed_dim, T=4)

    # ---- Mock SentenceTransformer ----
    def fake_sentence_transformer(name):
        # ensure the correct model name is passed (not strictly required)
        assert "all-MiniLM-L6-v2" in name
        return DummySentenceTransformer(dim=text_embed_dim)

    monkeypatch.setattr(sm, "SentenceTransformer", fake_sentence_transformer)

    # ---- Mock DiffusionModel imported in sampler.py ----
    monkeypatch.setattr(sm, "DiffusionModel", DummyDiffusionModel)

    # ---- Mock torch.load used inside sampler.sample ----
    # Return a dict with an empty 'model_state_dict' so access works
    def fake_torch_load(path):
        return {"model_state_dict": {}}

    monkeypatch.setattr(sm.torch, "load", fake_torch_load)

    # ---- Force CPU path so tests don't depend on CUDA ----
    monkeypatch.setattr(sm.torch.cuda, "is_available", lambda: False)

    # ---- Run the sampler ----
    z_t = sm.sample(cfg)

    # ---- Assertions ----
    assert isinstance(z_t, torch.Tensor)
    # shape should be [1, 8] as per your code
    assert z_t.shape == (1, 8)
    # values should be finite
    assert torch.isfinite(z_t).all()


def test_sample_raises_if_text_embed_dim_mismatch(monkeypatch):
    """
    If SentenceTransformer returns an embedding of the wrong dimension,
    the internal assert should fail.
    """

    # cfg expects text_embed_dim=32, but we'll give 16
    cfg = make_cfg(text_embed_dim=32, T=2)

    # Fake ST that returns dim=16
    def bad_sentence_transformer(name):
        return DummySentenceTransformer(dim=16)

    monkeypatch.setattr(sm, "SentenceTransformer", bad_sentence_transformer)
    monkeypatch.setattr(sm, "DiffusionModel", DummyDiffusionModel)
    monkeypatch.setattr(sm.torch, "load", lambda path: {"model_state_dict": {}})
    monkeypatch.setattr(sm.torch.cuda, "is_available", lambda: False)

    with pytest.raises(AssertionError):
        sm.sample(cfg)
