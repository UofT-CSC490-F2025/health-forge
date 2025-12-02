# test_sample.py

import numpy as np
import torch
import pytest

import final_project.sample as sm


# -----------------------
# Dummy / mock classes
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

    - Ignores cfg structure.
    - forward(z_t, text_embed, lambda_tensor) returns zeros with same shape as z_t.
    - load_state_dict / eval / to are no-ops.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, z_t, text_embed, lambda_tensor):
        # Predict zero noise to keep math simple/stable
        return torch.zeros_like(z_t)

    def load_state_dict(self, state_dict):
        # No-op (we don't care about weights here)
        return self

    def eval(self):
        return self

    def to(self, device=None):
        return self


class DummyAutoencoder(torch.nn.Module):
    """
    Minimal stand-in for EHRLatentAutoencoder.

    - __init__(input_dim, latent_dim) matches real signature
    - decoder: linear projection from latent_dim -> input_dim
    - load_state_dict, eval, to: no-ops / light behavior
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.decoder = torch.nn.Linear(latent_dim, input_dim)

    def load_state_dict(self, state_dict, strict=True):
        # Ignore contents; treat as loaded
        return self

    def eval(self):
        return self

    def to(self, device=None):
        return self


# -----------------------
# Helper to build a tiny cfg
# -----------------------

def make_cfg(text_embed_dim=16, T=3, input_dim=1024):
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
            "input_dim": input_dim,
            "text_embed_dim": text_embed_dim,
            # other keys are ignored by DummyDiffusionModel
        },
    }


# -----------------------
# Tests
# -----------------------

def test_sample_from_checkpoint_runs_and_returns_array(monkeypatch):
    """
    End-to-end-ish test of sample_from_checkpoint(cfg, ckpt):
    - Mocks SentenceTransformer to avoid network/model load
    - Mocks DiffusionModel so noise prediction is trivial
    - Mocks EHRLatentAutoencoder and np.load / torch.load
    - Forces CPU path (cuda.is_available=False)
    - Verifies returned array shape and finiteness
    """
    LATENT_DIM = 1024
    INPUT_DIM = 1806
    text_embed_dim = 16
    cfg = make_cfg(text_embed_dim=text_embed_dim, T=4, input_dim=LATENT_DIM)
    checkpoint_path = "dummy_diffusion_ckpt.pt"

    # ---- Mock SentenceTransformer ----
    def fake_sentence_transformer(name):
        # Ensure we're being called with the expected model name
        assert "embeddinggemma-300m-medical" in name
        return DummySentenceTransformer(dim=text_embed_dim)

    monkeypatch.setattr(sm, "SentenceTransformer", fake_sentence_transformer)

    # ---- Mock DiffusionModel imported in sample.py ----
    monkeypatch.setattr(sm, "DiffusionModel", DummyDiffusionModel)

    # ---- Mock EHRLatentAutoencoder imported in sample.py ----
    monkeypatch.setattr(sm, "EHRLatentAutoencoder", DummyAutoencoder)

    # ---- Mock torch.load so no real files / GPUs are used ----
    def fake_torch_load(path, map_location=None):
        # diffusion checkpoint
        if path == checkpoint_path:
            return {"model_state_dict": {}}
        # autoencoder checkpoint
        if path == "best_autoencoder_model.pt":
            # DummyAE.load_state_dict ignores contents anyway
            return {}
        raise AssertionError(f"Unexpected torch.load path: {path}, map_location={map_location}")

    monkeypatch.setattr(sm.torch, "load", fake_torch_load)

    # ---- Mock np.load for latent stats ----
    def fake_np_load(path):
        if path == "latent_mean.npy":
            return np.zeros(LATENT_DIM, dtype=np.float32)
        if path == "latent_std.npy":
            # non-zero std to avoid degeneracy, but harmless
            return np.ones(LATENT_DIM, dtype=np.float32)
        raise AssertionError(f"Unexpected np.load path: {path}")

    monkeypatch.setattr(sm.np, "load", fake_np_load)

    # ---- Force CPU path so tests don't depend on CUDA ----
    monkeypatch.setattr(sm.torch.cuda, "is_available", lambda: False)

    # ---- Run the sampler ----
    out = sm.sample_from_checkpoint(cfg, checkpoint_path, text_desc="simple test description")

    # ---- Assertions ----
    assert isinstance(out, np.ndarray)
    # sampler always creates 100 samples of dimension 1806
    assert out.shape == (100, INPUT_DIM)
    assert np.isfinite(out).all()


def test_sample_from_checkpoint_raises_if_text_embed_dim_mismatch(monkeypatch):
    """
    If SentenceTransformer returns an embedding of the wrong dimension,
    the internal assert `text_embed.shape[-1] == text_embed_dim` should fail.
    """

    LATENT_DIM = 1024
    # cfg expects text_embed_dim=32, but we'll give 16
    cfg = make_cfg(text_embed_dim=32, T=2, input_dim=LATENT_DIM)
    checkpoint_path = "dummy_diffusion_ckpt.pt"

    # Fake ST that returns dim=16 instead of 32
    def bad_sentence_transformer(name):
        return DummySentenceTransformer(dim=16)

    monkeypatch.setattr(sm, "SentenceTransformer", bad_sentence_transformer)
    monkeypatch.setattr(sm, "DiffusionModel", DummyDiffusionModel)
    monkeypatch.setattr(sm, "EHRLatentAutoencoder", DummyAutoencoder)

    # Still mock loads & CUDA so no real I/O occurs even if assert didn't fire early
    monkeypatch.setattr(
        sm.torch,
        "load",
        lambda path, map_location=None: {"model_state_dict": {}} if path == checkpoint_path else {},
    )

    def fake_np_load(path):
        return np.zeros(LATENT_DIM, dtype=np.float32)

    monkeypatch.setattr(sm.np, "load", fake_np_load)
    monkeypatch.setattr(sm.torch.cuda, "is_available", lambda: False)

    with pytest.raises(AssertionError):
        sm.sample_from_checkpoint(cfg, checkpoint_path, text_desc="whatever")
