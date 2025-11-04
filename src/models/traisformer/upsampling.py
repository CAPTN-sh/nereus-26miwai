import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsamplingDecoder(nn.Module):
    def __init__(self, n_embd, x_size, y_size, C=4, min_seed=64, max_levels=4):
        super().__init__()
        self.x_size, self.y_size = x_size, y_size
        n_levels = min(
            max_levels,
            max(0, math.ceil(math.log2(max(x_size, y_size) / min_seed))),
        )
        self.x_seed = math.ceil(x_size / (2**n_levels))
        self.y_seed = math.ceil(y_size / (2**n_levels))

        self.z2seed = nn.Linear(n_embd, self.x_seed * self.y_seed)
        self.stem = nn.Conv2d(1, C, 3, padding=1, bias=True)

        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2) for _ in range(n_levels)])
        self.convs = nn.ModuleList(
            [nn.Conv2d(C, C, 3, padding=1) for _ in range(n_levels)]
        )
        self.films = nn.ModuleList([nn.Linear(n_embd, 2 * C) for _ in range(n_levels)])
        self.head = nn.Conv2d(C, 1, 1)

    def forward(self, z):
        z = z.squeeze(1)
        B = z.size(0)
        seed = self.z2seed(z).view(B, 1, self.x_seed, self.y_seed)
        x = F.relu(self.stem(seed))
        for up, conv, film in zip(self.ups, self.convs, self.films):
            x = up(x)
            x = F.silu(conv(x))
            g, b = film(z).chunk(2, dim=-1)  # (B,C),(B,C)
            x = x * (1 + torch.tanh(g))[:, :, None, None] + b[:, :, None, None]

        logits = self.head(x)
        logits = logits[:, :, : self.x_size, : self.y_size]
        return logits
