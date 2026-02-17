from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import DAEParams

class Block(nn.Module):
    """Ein Transformer-Block"""

    def __init__(self, config: DAEParams):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
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