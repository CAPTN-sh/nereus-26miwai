"""Causal transformer block for the autoregressive TrAISformer.

The heatmap variant's :class:`..modules.transformer_block.Block` is bidirectional -- it
pools the whole observation window into one context token, so it has no reason to mask
the future. Next-token prediction does, hence a separate block.
"""
import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = config.attn_dropout
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, attn_mask, cache=None):
        """``attn_mask`` bool, True where a query may attend to a key.

        With ``cache`` (a dict mutated in place) the keys and values of previous calls
        are reused, so a rollout costs one token per step instead of re-encoding the
        whole prefix 30 times.
        """
        B, L, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, L, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        return self.resid_drop(self.proj(y))


class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, attn_mask, cache=None):
        x = x + self.attn(self.ln1(x), attn_mask, cache)
        return x + self.mlp(self.ln2(x))


def causal_padding_mask(key_valid):
    """Combine a causal mask with per-key validity. ``key_valid`` [B, L] -> [B, 1, L, L].

    Trajectories are left-padded, so a plain causal mask would let real timesteps attend
    to padding tokens (which discretise to a fixed corner cell). Padded *queries* are
    given their own diagonal back: without it a fully-masked row makes softmax produce
    NaN, and those rows are dropped by the loss mask anyway.
    """
    B, L = key_valid.shape
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=key_valid.device))
    allowed = causal.unsqueeze(0) & key_valid[:, None, :]
    allowed = allowed | torch.eye(L, dtype=torch.bool, device=key_valid.device).unsqueeze(0)
    return allowed.unsqueeze(1)
