import math
import numpy as np
import torch
import pytest

import final_project.data_utils as dd


def test_diffusion_dataset_len():
    data = torch.randn(10, 4)
    text_embeds = torch.randn(10, 6)
    ds = dd.DiffusionDataset(
        data=data,
        text_embeds=text_embeds,
        T=100,
        noise_a=0.1,
        noise_b=0.2,
        embed_drop_prob=0.3,
        device="cpu",
    )
    assert len(ds) == 10


def test_diffusion_dataset_getitem_shapes():
    B, D, E = 8, 4, 6
    data = torch.randn(B, D)
    text_embeds = torch.randn(B, E)

    ds = dd.DiffusionDataset(
        data=data,
        text_embeds=text_embeds,
        T=100,
        noise_a=0.1,
        noise_b=0.2,
        embed_drop_prob=0.3,
        device="cpu",
    )

    z_l, text_embed, lambda_scaled, epsilon = ds[0]

    assert isinstance(z_l, torch.Tensor)
    assert isinstance(text_embed, torch.Tensor)
    assert isinstance(epsilon, torch.Tensor)

    assert z_l.shape == (D,)
    assert text_embed.shape == (E,)
    assert epsilon.shape == (D,)

    # lambda_scaled is a Python float for a single item
    assert isinstance(lambda_scaled, float)
    assert -1.0 <= lambda_scaled <= 1.0


def test_diffusion_dataset_embed_not_dropped_when_prob_zero():
    data = torch.randn(3, 4)
    text_embeds = torch.randn(3, 6)

    ds = dd.DiffusionDataset(
        data=data,
        text_embeds=text_embeds.clone(),
        T=100,
        noise_a=0.1,
        noise_b=0.2,
        embed_drop_prob=0.0,  # never drop
        device="cpu",
    )

    _, text_embed, _, _ = ds[1]
    # Should exactly equal the original embedding
    assert torch.allclose(text_embed, text_embeds[1])


def test_diffusion_dataset_embed_always_dropped_when_prob_one(monkeypatch):
    data = torch.randn(3, 4)
    text_embeds = torch.randn(3, 6)

    # Make random.random always return 0.0 so it's < 1.0
    monkeypatch.setattr(dd.random, "random", lambda: 0.0)

    ds = dd.DiffusionDataset(
        data=data,
        text_embeds=text_embeds,
        T=100,
        noise_a=0.1,
        noise_b=0.2,
        embed_drop_prob=1.0,  # always drop
        device="cpu",
    )

    _, text_embed, _, _ = ds[0]
    # Should be all zeros
    assert torch.all(text_embed == 0)


def test_diffusion_dataset_noise_combination(monkeypatch):
    """
    Verify that z_l = signal_coeff * x + noise_coeff * epsilon
    when we control the randomness.
    """
    # Simple deterministic data
    x_vec = torch.ones(4)  # [1, 1, 1, 1]
    data = x_vec.unsqueeze(0)  # shape [1, 4]
    text_embeds = torch.randn(1, 6)

    noise_a = 0.3
    noise_b = 0.5

    ds = dd.DiffusionDataset(
        data=data,
        text_embeds=text_embeds,
        T=100,
        noise_a=noise_a,
        noise_b=noise_b,
        embed_drop_prob=0.0,  # never drop embeds
        device="cpu",
    )

    # Fix u = 0.5 for the logistic-like schedule
    def fake_random():
        return 0.5

    # Fix epsilon = [2, 2, 2, 2] for simplicity
    def fake_randn_like(x):
        return torch.full_like(x, 2.0)

    monkeypatch.setattr(dd.random, "random", fake_random)
    monkeypatch.setattr(dd.torch, "randn_like", fake_randn_like)

    z_l, text_embed, lambda_scaled, epsilon = ds[0]

    # Check epsilon is exactly what we expect
    assert torch.allclose(epsilon, torch.full_like(x_vec, 2.0))

    # Manually compute expected coefficients
    u = 0.5
    l_val = -2 * math.log(math.tan(noise_a * u + noise_b))
    signal_coeff = math.sqrt(1 / (1 + math.exp(-l_val)))
    noise_coeff = math.sqrt(1 - signal_coeff ** 2)

    expected_z = signal_coeff * x_vec + noise_coeff * epsilon

    assert torch.allclose(z_l, expected_z, atol=1e-6)
    # lambda_scaled = tanh(l / 40)
    expected_lambda_scaled = math.tanh(l_val / 40.0)
    assert abs(lambda_scaled - expected_lambda_scaled) < 1e-6


def test_prepare_diffusion_dataloaders_torch_input():
    # Small synthetic dataset
    N, D, E = 20, 4, 6
    data = torch.randn(N, D)
    text_embeds = torch.randn(N, E)

    cfg = {
        "test_split": 0.2,   # 4 test, 16 train
        "T": 100,
        "num_workers": 0,    # keep 0 for tests
        "batch_size": 5,
        "lambda_min": -1.0,
        "lambda_max": 1.0,
        "embed_drop_prob": 0.1,
    }

    device = "cpu"

    train_loader, test_loader = dd.prepare_diffusion_dataloaders(
        data=data,
        text_embeds=text_embeds,
        cfg=cfg,
        device=device,
    )

    # Check dataloader types
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)

    # Check dataset lengths (16 train, 4 test)
    assert len(train_loader.dataset) + len(test_loader.dataset) == N
    assert len(test_loader.dataset) == int(N * cfg["test_split"])

    # Check batch shapes
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    z_l_train, text_embed_train, lambda_train, eps_train = train_batch
    z_l_test, text_embed_test, lambda_test, eps_test = test_batch

    # First dimension = batch size (except possibly last batch, so we only check <=)
    assert z_l_train.shape[0] <= cfg["batch_size"]
    assert text_embed_train.shape[0] == z_l_train.shape[0]
    assert eps_train.shape[0] == z_l_train.shape[0]
    # lambda_scaled should be a 1D tensor of length batch_size
    assert isinstance(lambda_train, torch.Tensor)
    assert lambda_train.shape[0] == z_l_train.shape[0]
    assert lambda_train.dim() == 1

    assert z_l_test.shape[0] <= cfg["batch_size"]
    assert text_embed_test.shape[0] == z_l_test.shape[0]
    assert eps_test.shape[0] == z_l_test.shape[0]
    assert isinstance(lambda_test, torch.Tensor)
    assert lambda_test.shape[0] == z_l_test.shape[0]
    assert lambda_test.dim() == 1

    # Values of lambda_scaled should be in [-1, 1] due to tanh
    assert torch.all(lambda_train >= -1.0) and torch.all(lambda_train <= 1.0)
    assert torch.all(lambda_test >= -1.0) and torch.all(lambda_test <= 1.0)

    # pin_memory is always True in prepare_diffusion_dataloaders
    assert train_loader.pin_memory is True
    assert test_loader.pin_memory is True

    # embed_drop_prob is passed correctly into underlying dataset
    assert train_loader.dataset.embed_drop_prob == cfg["embed_drop_prob"]
    assert test_loader.dataset.embed_drop_prob == cfg["embed_drop_prob"]
