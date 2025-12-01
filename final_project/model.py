import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.functional import softmax
from torch.nn import Linear, Sequential, SiLU, LayerNorm, Dropout, Module, RMSNorm, Parameter

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
        
        self.lambda_mlp = Sequential(
            Linear(1, hidden_dim),
            SiLU(),
            Linear(hidden_dim, hidden_dim)
        )

        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim)
        self.layer_norm = RMSNorm(hidden_dim)

        
        # Main blocks
        self.blocks = torch.nn.ModuleList()  # Use ModuleList instead of []
        for _ in range(num_layers):
            # self.blocks.append(TransformerBlock(hidden_dim, num_heads, dropout))
            # self.blocks.append(MLPBlock(hidden_dim, text_embed_dim, dropout))
            self.blocks.append(GLUBlock(hidden_dim, text_embed_dim, dropout))

        
        # Output projection
        self.output = Sequential(
            RMSNorm(hidden_dim),
            Linear(hidden_dim, input_dim)
        )
    
    
    def forward(self, x, text_embed, lambda_val):
        # x: [B, D] - batch of 1D vectors
        # text_embed: [B, T] - Batch of 1D vectors for text embeddings
                
        # Project input
        h = self.input_proj(x)  # [B, hidden_dim]
        lambda_embed = self.lambda_mlp(lambda_val)
        h = h + lambda_embed
        h = self.layer_norm(h)


        
        # Process through blocks
        for block in self.blocks:
            h = block(h, text_embed)  # Can optionally pass t_emb to blocks
        
        return self.output(h)

class GLUBlock(Module):
    def __init__(self, dim, text_embed_dim, dropout):
        super().__init__()
        # First GLU MLP
        self.norm1 = RMSNorm(dim)
        self.fc1 = Linear(dim, 4 * dim)
        self.fc_gate1 = Linear(dim, 4 * dim)
        self.fc_out1 = Linear(4 * dim, dim)
        self.dropout1 = Dropout(dropout)
        self.res_scale1 = Parameter(torch.tensor(1.0))

        # Cross-attention on text embeddings
        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(text_embed_dim, dim)
        self.v_proj = Linear(text_embed_dim, dim)
        self.attn_norm = RMSNorm(dim)

        # Second GLU MLP
        self.norm2 = RMSNorm(dim)
        self.fc2 = Linear(dim, 4 * dim)
        self.fc_gate2 = Linear(dim, 4 * dim)
        self.fc_out2 = Linear(4 * dim, dim)
        self.dropout2 = Dropout(dropout)
        self.res_scale2 = Parameter(torch.tensor(1.0))

    def forward(self, x, text_embed):
        # ----- 1) First GLU MLP -----
        h = self.norm1(x)
        gate1 = torch.sigmoid(self.fc_gate1(h))
        h = self.fc1(h) * gate1          # GLU
        h = self.fc_out1(h)
        h = self.dropout1(h)
        x = x + self.res_scale1 * h      # residual

        # ----- 2) Cross-attention with text conditioning -----
        q = self.q_proj(x)               # [B, D]
        k = self.k_proj(text_embed)      # [B, T, D]
        v = self.v_proj(text_embed)      # [B, T, D]

        attn_out = scaled_dot_product_attention(q, k, v)
        x = x + attn_out
        x = self.attn_norm(x)

        # ----- 3) Second GLU MLP -----
        h = self.norm2(x)
        gate2 = torch.sigmoid(self.fc_gate2(h))
        h = self.fc2(h) * gate2
        h = self.fc_out2(h)
        h = self.dropout2(h)
        x = x + self.res_scale2 * h

        return x

class MLPBlock(Module):
    def __init__(self, dim, text_embed_dim, dropout):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mlp1 = Sequential(
            Linear(dim, 4 * dim),
            SiLU(),
            Dropout(dropout),
            Linear(4 * dim, dim)
        )

        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(text_embed_dim, dim)
        self.v_proj = Linear(text_embed_dim, dim)

        self.attn_norm = RMSNorm(dim)
        
        self.norm2 = RMSNorm(dim)
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
        
    
    def forward(self, x):
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



