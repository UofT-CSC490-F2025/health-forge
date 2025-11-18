import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from torch.nn import Linear, Sequential, SiLU, LayerNorm, Dropout, Module
import yaml
import argparse

class DiffusionModel(Module):
    def __init__(self, cfg):     # Use transformer vs MLP
        super().__init__()
        input_dim = cfg["input_dim"]
        hidden_dim = cfg["hidden_dim"]
        time_emb_dim = cfg["time_emb_dim"]
        num_layers = cfg["num_layers"]
        num_heads = cfg["num_heads"]
        dropout = cfg["dropout"]
        use_attention = cfg["use_attention"]


        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim)
        self.time_emb_dim = time_emb_dim
        # Time embedding
        self.time_embed = Sequential(
            Linear(time_emb_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, hidden_dim)
        )
        
        # Main blocks
        self.blocks = []
        for _ in range(num_layers):
            if use_attention:
                self.blocks.append(TransformerBlock(hidden_dim, num_heads, dropout))
            else:
                self.blocks.append(MLPBlock(hidden_dim, hidden_dim, dropout))
        
        # Output projection
        self.output = Sequential(
            LayerNorm(hidden_dim),
            Linear(hidden_dim, input_dim)
        )
    
    @staticmethod
    def timestep_embedding(t, dim):
        # Sinusoidal positional encoding
        half_dim = dim // 2
        emb = 10000 / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, t):
        # x: [B, D] - batch of 1D vectors
        # t: [B] - timesteps
        
        # Time embedding
        t_emb = self.timestep_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]
        t_emb = self.time_embed(t_emb)  # [B, hidden_dim]
        
        # Project input
        h = self.input_proj(x)  # [B, hidden_dim]
        
        # Process through blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output
        return self.output(h)  # [B, input_dim]


class MLPBlock(Module):
    def __init__(self, dim, time_emb_dim, dropout):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.mlp1 = Sequential(
            Linear(dim, 4 * dim),
            SiLU(),
            Dropout(dropout),
            Linear(4 * dim, dim)
        )
        
        self.time_proj = Sequential(
            SiLU(),
            Linear(time_emb_dim, dim)
        )
        
        self.norm2 = LayerNorm(dim)
        self.mlp2 = Sequential(
            Linear(dim, 4 * dim),
            SiLU(),
            Dropout(dropout),
            Linear(4 * dim, dim)
        )
    
    def forward(self, x, t_emb):
        # First block with time injection
        h = self.norm1(x)
        h = self.mlp1(h) + self.time_proj(t_emb)
        x = x + h
        
        # Second block
        h = self.norm2(x)
        h = self.mlp2(h)
        return x + h


class TransformerBlock(Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        
        self.norm2 = LayerNorm(dim)
        self.mlp = Sequential(
            Linear(dim, 4 * dim),
            SiLU(),
            Dropout(dropout),
            Linear(4 * dim, dim),
            Dropout(dropout)
        )
        
        self.time_proj = Sequential(
            SiLU(),
            Linear(dim, dim)
        )
    
    def forward(self, x, t_emb):
        # Self-attention with time modulation
        h = self.norm1(x)
        h = self.attn(h) + self.time_proj(t_emb)
        x = x + h
        
        # MLP
        h = self.norm2(x)
        h = self.mlp(h)
        return x + h


class MultiHeadAttention(Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        # x: [B, D]
        B, D = x.shape
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [B, 1, D]
        
        qkv = self.qkv(x).reshape(B, 1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [B, num_heads, 1, head_dim]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, 1, 1]
        attn = softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)
        out = self.proj(out).squeeze(1)  # [B, D]
        
        return out



