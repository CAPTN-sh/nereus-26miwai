import torch
from torch import nn


class LinearHead(nn.Module):
    # Approach 0
    def __init__(self, n_embd, x_size, y_size, k_rank = None):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size

        self.intent_head = nn.Linear(n_embd, x_size * y_size)

    def forward(self, z):
        B = z.shape[0]
        heatmap = self.intent_head(z)
        return heatmap.view(B, 1, self.x_size, self.y_size)

class LowRankHead(nn.Module):
    # Approach 3
    def __init__(self, n_embd, x_size, y_size, k_rank = 16):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size

        self.rank_proj = nn.Linear(n_embd, k_rank)
        self.intent_head = nn.Linear(k_rank, x_size * y_size)

    def forward(self, z):
        B = z.shape[0]
        rank = self.rank_proj(z)
        heatmap = self.intent_head(rank)
        return heatmap.view(B, 1, self.x_size, self.y_size)

class FactorizedHead(nn.Module):
    # Approach 1
    def __init__(self, n_embd, x_size, y_size, k_rank = 8):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.r = k_rank

        self.x_intent = nn.Linear(n_embd, k_rank * x_size)
        self.y_intent = nn.Linear(n_embd, k_rank * y_size)

    def forward(self, z):
        B = z.shape[0]
        a = self.x_intent(z).view(B, self.r, self.x_size, 1)
        b = self.y_intent(z).view(B, self.r, 1, self.y_size)
        heatmap = torch.sum(a * b, dim=1)
        return heatmap.view(B, 1, self.x_size, self.y_size)

class CNNHead(nn.Module):
    # Approach 2
    def __init__(self, n_embd, x_size, y_size, k_rank = 30):
        super().__init__()
        self.h, self.w = int(x_size * k_rank/100), int(y_size * k_rank/100)

        self.proj = nn.Linear(n_embd, self.h * self.w)
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Upsample(size=(x_size, y_size), mode="bilinear", align_corners=False),
            nn.Conv2d(1, 1, 3, padding=1),
        )

    def forward(self, z):
        B = z.shape[0]
        x = self.proj(z).view(B, -1, self.h, self.w)
        heatmap = self.decoder(x)
        return heatmap

class MixtureHead(nn.Module):
    # Approach 4
    def __init__(self, n_embd, x_size, y_size, k_rank = 32):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size

        self.heatmaps = nn.Parameter(torch.randn(k_rank, x_size, y_size) * 0.1)
        self.importance = nn.Linear(n_embd, k_rank)

    def forward(self, z):
        B = z.shape[0]
        weights = self.importance(z)
        heatmap = torch.einsum("bk,kxy->bxy", weights, self.heatmaps)
        return heatmap.unsqueeze(1)
