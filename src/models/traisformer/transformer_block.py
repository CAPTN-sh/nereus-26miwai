from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import TraisformerParams

class Block(nn.Module):
    """Ein Transformer-Block"""

    def __init__(self, config: TraisformerParams):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.attn_dropout,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.attn_dropout),
        )

    def forward(self, x, mask):
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(
            query=x, 
            key=x, 
            value=x, 
            key_padding_mask=~mask
        )
        
        x = residual + attn_out
        x = x + self.mlp(self.ln2(x))
        return x