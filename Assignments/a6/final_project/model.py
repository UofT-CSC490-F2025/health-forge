import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.functional import softmax
from torch.nn import Linear, Sequential, SiLU, LayerNorm, Dropout, Module

import math 

class DiffusionModel(Module):
    def __init__(self, cfg):     # Use transformer vs MLP
        super().__init__()
        input_dim = cfg["input_dim"]
        hidden_dim = cfg["hidden_dim"]
        text_embed_dim = cfg["text_embed_dim"]
        num_layers = cfg["num_layers"]
        num_heads = cfg["num_heads"]
        dropout = cfg["dropout"]
        use_attention = cfg["use_attention"]


        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim)
        
        # Main blocks
        self.blocks = torch.nn.ModuleList()  # Use ModuleList instead of []
        for _ in range(num_layers):
            if use_attention:
                self.blocks.append(TransformerBlock(hidden_dim, num_heads, dropout))
            else:
                self.blocks.append(MLPBlock(hidden_dim, text_embed_dim, dropout))
        
        # Output projection
        self.output = Sequential(
            LayerNorm(hidden_dim),
            Linear(hidden_dim, input_dim)
        )
    
    
    def forward(self, x, text_embed):
        # x: [B, D] - batch of 1D vectors
        # text_embed: [B, T] - Batch of 1D vectors for text embeddings
                
        # Project input
        h = self.input_proj(x)  # [B, hidden_dim]
        
        # Process through blocks
        for block in self.blocks:
            # TODO: CROSS ATTENTION WITH TEXT EMBEDDING
            h = block(h, text_embed)
        
        # Output
        return self.output(h)  # [B, input_dim]


class MLPBlock(Module):
    def __init__(self, dim, text_embed_dim, dropout):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.mlp1 = Sequential(
            Linear(dim, 4 * dim),
            SiLU(),
            Dropout(dropout),
            Linear(4 * dim, dim)
        )

        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(text_embed_dim, dim)
        self.v_proj = Linear(text_embed_dim, dim)

        self.attn_norm = LayerNorm(dim)
        
        self.norm2 = LayerNorm(dim)
        self.mlp2 = Sequential(
            Linear(dim, 4 * dim),
            SiLU(),
            Dropout(dropout),
            Linear(4 * dim, dim)
        )
    
    def forward(self, x, text_embed):
        h = self.norm1(x)
        h = self.mlp1(h)
        x = x + h # residual connection

        q = self.q_proj(x)
        k = self.k_proj(text_embed)
        v = self.v_proj(text_embed)

        attn_scores = scaled_dot_product_attention(q, k, v)
        x = x + attn_scores
        x = self.attn_norm(x)

        
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
        
    
    def forward(self, x, text_embed):
        # Self-attention with time modulation
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h
        
        # MLP
        h = self.norm2(x)
        h = self.mlp(h)
        return x + h


class MultiHeadAttention(Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()

        if dim%num_heads > 0:
            raise ValueError("Total dimension not divisible by heads")

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



