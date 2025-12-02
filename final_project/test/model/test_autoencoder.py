import torch
import torch.nn as nn
import pytest

from final_project.autoencoder import Encoder, Decoder, EHRLatentAutoencoder


# -----------------------
# Helper config
# -----------------------

def make_dims():
    # Use smaller dims than full 1804 for fast tests, but keep structure
    return {
        "input_dim": 64,
        "latent_dim": 32,
    }


# -----------------------
# Encoder tests
# -----------------------

def test_encoder_output_shape_and_finiteness():
    dims = make_dims()
    enc = Encoder(input_dim=dims["input_dim"], latent_dim=dims["latent_dim"])

    B = 10
    x = torch.randn(B, dims["input_dim"])

    z = enc(x)

    assert isinstance(z, torch.Tensor)
    assert z.shape == (B, dims["latent_dim"])
    assert torch.isfinite(z).all()


def test_encoder_backprop():
    dims = make_dims()
    enc = Encoder(input_dim=dims["input_dim"], latent_dim=dims["latent_dim"])

    B = 8
    x = torch.randn(B, dims["input_dim"], requires_grad=True)

    z = enc(x)
    loss = z.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# -----------------------
# Decoder tests
# -----------------------

def test_decoder_output_shape_and_finiteness():
    dims = make_dims()
    dec = Decoder(latent_dim=dims["latent_dim"], output_dim=dims["input_dim"])

    B = 7
    z = torch.randn(B, dims["latent_dim"])

    logits = dec(z)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (B, dims["input_dim"])
    assert torch.isfinite(logits).all()


def test_decoder_backprop():
    dims = make_dims()
    dec = Decoder(latent_dim=dims["latent_dim"], output_dim=dims["input_dim"])

    B = 5
    z = torch.randn(B, dims["latent_dim"], requires_grad=True)

    logits = dec(z)
    loss = logits.abs().mean()
    loss.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


# -----------------------
# Autoencoder tests
# -----------------------

def test_autoencoder_forward_shapes_and_finiteness():
    dims = make_dims()
    ae = EHRLatentAutoencoder(
        input_dim=dims["input_dim"],
        latent_dim=dims["latent_dim"],
    )

    B = 6
    x = torch.randn(B, dims["input_dim"])

    logits, z = ae(x)

    # Shape checks
    assert isinstance(logits, torch.Tensor)
    assert isinstance(z, torch.Tensor)
    assert logits.shape == (B, dims["input_dim"])
    assert z.shape == (B, dims["latent_dim"])

    # Finite checks
    assert torch.isfinite(logits).all()
    assert torch.isfinite(z).all()


def test_autoencoder_backprop_end_to_end():
    dims = make_dims()
    ae = EHRLatentAutoencoder(
        input_dim=dims["input_dim"],
        latent_dim=dims["latent_dim"],
    )

    B = 4
    x = torch.randn(B, dims["input_dim"], requires_grad=False)

    # Simple MSE reconstruction loss using logits directly
    logits, z = ae(x)
    loss = nn.MSELoss()(logits, x)

    loss.backward()

    # Check some parameter got a gradient
    grads = [p.grad for p in ae.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_autoencoder_single_optim_step_changes_parameters():
    dims = make_dims()
    ae = EHRLatentAutoencoder(
        input_dim=dims["input_dim"],
        latent_dim=dims["latent_dim"],
    )

    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)

    B = 8
    x = torch.randn(B, dims["input_dim"])

    # Snapshot parameters before update
    before_params = [p.detach().clone() for p in ae.parameters()]

    logits, z = ae(x)
    loss = nn.MSELoss()(logits, x)

    opt.zero_grad()
    loss.backward()
    opt.step()

    # Compare after update
    after_params = list(ae.parameters())
    changed = any(
        not torch.allclose(b, a.detach(), atol=1e-7)
        for b, a in zip(before_params, after_params)
    )

    assert changed, "Expected at least one autoencoder parameter to change after optimizer step."
