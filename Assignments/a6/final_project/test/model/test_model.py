import torch
import pytest

import final_project.model as dm  


def make_cfg(use_attention: bool):
    return {
        "input_dim": 8,
        "hidden_dim": 16,
        "text_embed_dim": 12,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "use_attention": use_attention,
    }


# ------------------------
# DiffusionModel (MLP path)
# ------------------------

def test_diffusion_model_mlp_forward_shape_and_finiteness():
    cfg = make_cfg(use_attention=False)
    model = dm.DiffusionModel(cfg)

    B = 5
    x = torch.randn(B, cfg["input_dim"])
    text_embed = torch.randn(B, cfg["text_embed_dim"])

    out = model(x, text_embed)

    assert out.shape == (B, cfg["input_dim"])
    assert torch.isfinite(out).all()


def test_diffusion_model_mlp_backprop():
    cfg = make_cfg(use_attention=False)
    model = dm.DiffusionModel(cfg)

    B = 4
    x = torch.randn(B, cfg["input_dim"], requires_grad=True)
    text_embed = torch.randn(B, cfg["text_embed_dim"])

    out = model(x, text_embed)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ----------------------------
# DiffusionModel (attention path)
# ----------------------------

def test_diffusion_model_attention_forward_shape_and_finiteness():
    cfg = make_cfg(use_attention=False)
    model = dm.DiffusionModel(cfg)

    B = 3
    x = torch.randn(B, cfg["input_dim"])
    text_embed = torch.randn(B, cfg["text_embed_dim"])

    out = model(x, text_embed)

    assert out.shape == (B, cfg["input_dim"])
    assert torch.isfinite(out).all()


def test_diffusion_model_attention_backprop():
    cfg = make_cfg(use_attention=False)
    model = dm.DiffusionModel(cfg)

    B = 2
    x = torch.randn(B, cfg["input_dim"], requires_grad=True)
    text_embed = torch.randn(B, cfg["text_embed_dim"])

    out = model(x, text_embed)
    loss = out.abs().sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# -----------------
# MLPBlock specific
# -----------------

def test_mlpblock_output_shape():
    dim = 16
    text_dim = 12
    dropout = 0.0

    block = dm.MLPBlock(dim, text_dim, dropout)

    B = 7
    x = torch.randn(B, dim)
    text_embed = torch.randn(B, text_dim)

    out = block(x, text_embed)

    assert out.shape == (B, dim)
    assert torch.isfinite(out).all()



# -------------------------
# MultiHeadAttention shapes
# -------------------------

def test_multihead_attention_shape():
    dim = 16
    num_heads = 4
    dropout = 0.0
    B = 6

    attn = dm.MultiHeadAttention(dim, num_heads, dropout)
    x = torch.randn(B, dim)

    out = attn(x)

    assert out.shape == (B, dim)
    assert torch.isfinite(out).all()


def test_multihead_attention_requires_divisible_dim():
 
    with pytest.raises(ValueError):
        dim = 10
        num_heads = 4  # 10 // 4 = 2 (remainder 2) -> janky but will still "run"
        dropout = 0.0

        dm.MultiHeadAttention(dim, num_heads, dropout)
    



# ------------------
# TransformerBlock
# ------------------

def test_transformerblock_shape_and_grad():
    dim = 16
    num_heads = 4
    dropout = 0.1
    B = 5

    block = dm.TransformerBlock(dim, num_heads, dropout)

    x = torch.randn(B, dim, requires_grad=True)

    # because we modified forward to accept text_embed=None
    out = block(x, text_embed=None)

    assert out.shape == (B, dim)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
