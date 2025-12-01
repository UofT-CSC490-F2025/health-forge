import math
import torch
import torch.nn.functional as F
from torch.nn import (
    Linear,
    Sequential,
    SiLU,
    LayerNorm,
    Dropout,
    Module,
    GELU,
)


class DiffusionModel(Module):
    def __init__(self, cfg):
        """
        cfg:
          input_dim: int
          hidden_dim: int
          text_embed_dim: int
          num_layers: int
          num_heads: int
          dropout: float
          use_attention: bool
        """
        super().__init__()
        input_dim = cfg["input_dim"]
        hidden_dim = cfg["hidden_dim"]
        text_embed_dim = cfg["text_embed_dim"]
        num_layers = cfg["num_layers"]
        num_heads = cfg["num_heads"]
        dropout = cfg["dropout"]
        use_attention = cfg["use_attention"]

        self.use_attention = use_attention

        # Encode lambda (noise level)
        self.lambda_mlp = Sequential(
            Linear(1, hidden_dim),
            SiLU(),
            Linear(hidden_dim, hidden_dim),
        )

        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim)
        self.layer_norm = LayerNorm(hidden_dim)

        # Main blocks
        self.blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            if use_attention:
                self.blocks.append(
                    TransformerBlock(
                        dim=hidden_dim,
                        context_dim=text_embed_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                )
            else:
                self.blocks.append(
                    MLPBlock(
                        dim=hidden_dim,
                        text_embed_dim=text_embed_dim,
                        dropout=dropout,
                    )
                )

        # Output projection
        self.output = Sequential(
            LayerNorm(hidden_dim),
            Linear(hidden_dim, input_dim),
        )

    def forward(self, x, text_embed, lambda_val):
        """
        x:          [B, D]           latent vector
        text_embed: [B, T] or [B, text_embed_dim]
        lambda_val: [B, 1]
        """
        # Project input
        h = self.input_proj(x)  # [B, hidden_dim]

        # Inject lambda
        lambda_embed = self.lambda_mlp(lambda_val)  # [B, hidden_dim]
        h = h + lambda_embed
        h = self.layer_norm(h)

        # Process through blocks
        for block in self.blocks:
            h = block(h, text_embed)

        return self.output(h)


# ---------------------------
# MLP-only block (no attention)
# ---------------------------
class MLPBlock(Module):
    def __init__(self, dim, text_embed_dim, dropout):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.text_proj = Linear(text_embed_dim, dim)

        self.mlp1 = Sequential(
            Linear(dim, 4 * dim),
            SiLU(),
            Dropout(dropout),
            Linear(4 * dim, dim),
            Dropout(dropout),
        )

        self.norm2 = LayerNorm(dim)
        self.mlp2 = Sequential(
            Linear(dim, 4 * dim),
            SiLU(),
            Dropout(dropout),
            Linear(4 * dim, dim),
            Dropout(dropout),
        )

    def forward(self, x, text_embed):
        """
        x:          [B, dim]
        text_embed: [B, text_embed_dim]
        """
        # First MLP with text conditioning (FiLM-style add)
        h = self.norm1(x)
        cond = self.text_proj(text_embed)  # [B, dim]
        h = h + cond
        h = self.mlp1(h)
        x = x + h  # residual

        # Second MLP
        h2 = self.norm2(x)
        h2 = self.mlp2(h2)
        x = x + h2

        return x


# ---------------------------
# Transformer-style block
# ---------------------------
class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = Sequential(
            Linear(dim, dim * mult),
            GELU(),
            Dropout(dropout),
            Linear(dim * mult, dim),
            Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(Module):
    def __init__(self, dim, context_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn1 = MultiHeadAttention(dim, num_heads, dropout)

        self.norm2 = LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, num_heads, dropout)

        self.norm3 = LayerNorm(dim)
        self.ff = FeedForward(dim, mult=4, dropout=dropout)

    def forward(self, x, context):
        """
        x:       [B, dim]
        context: [B, context_dim]  (e.g. sentence embedding)
        """
        # Self-attention (on x)
        x = x + self.attn1(self.norm1(x))

        # Cross-attention (x queries, context keys/values)
        x = x + self.cross_attn(self.norm2(x), context)

        # Feedforward
        x = x + self.ff(self.norm3(x))

        return x


# ---------------------------
# Multi-head self-attention
# ---------------------------
class MultiHeadAttention(Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        """
        x: [B, dim]
        We treat this as a sequence of length 1: [B, 1, dim]
        """
        B, D = x.shape
        x_seq = x.unsqueeze(1)  # [B, 1, D]

        qkv = self.qkv(x_seq)  # [B, 1, 3D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, 1, D]

        # Reshape for heads: [B, heads, 1, head_dim]
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, 1, 1]
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)
        out = self.proj(out).squeeze(1)  # [B, D]

        return out


# ---------------------------
# Cross-attention
# ---------------------------
class CrossAttention(Module):
    def __init__(self, dim, context_dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = Linear(dim, dim)
        self.to_k = Linear(context_dim, dim)
        self.to_v = Linear(context_dim, dim)

        self.out_proj = Linear(dim, dim)
        self.dropout = Dropout(dropout)

    def forward(self, x, context):
        """
        x:       [B, dim]              (latent)
        context: [B, context_dim]      (text embedding)
        """
        B, D = x.shape

        # Treat both as length-1 sequences
        x_seq = x.unsqueeze(1)          # [B, 1, D]
        context_seq = context.unsqueeze(1)  # [B, 1, Cdim]

        q = self.to_q(x_seq)           # [B, 1, D]
        k = self.to_k(context_seq)     # [B, 1, D]
        v = self.to_v(context_seq)     # [B, 1, D]

        # reshape for heads
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, 1, 1]
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)
        out = self.out_proj(out).squeeze(1)  # [B, D]

        return out
