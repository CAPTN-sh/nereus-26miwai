"""ST-GCNN and TXP-CNN blocks, ported from Social-STGCNN (Mohamed et al., 2020).

IS-STGCNN reuses these unchanged -- its contributions are the adjacency kernel, the
social sampling and the MPC correction, all of which live outside this file.
"""
import torch
from torch import nn


class ConvTemporalGraphical(nn.Module):
    """1x1 feature transform followed by aggregation over the normalised kernel."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, a):
        """x [B, C_in, T, N], a [B, T, N, N] -> [B, C_out, T, N]."""
        x = self.conv(x)
        return torch.einsum("nctv,ntvw->nctw", x, a).contiguous()


class STGCNBlock(nn.Module):
    """Spatial graph conv + temporal conv with a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        assert kernel_size % 2 == 1, "temporal kernel must be odd for 'same' padding"
        padding = ((kernel_size - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        self.prelu = nn.PReLU()

    def forward(self, x, a):
        res = self.residual(x)
        x = self.tcn(self.gcn(x, a)) + res
        return self.prelu(x)


class TXPCNN(nn.Module):
    """Time-extrapolator CNN: convolves over the *time* axis as channels.

    Input [B, C, T_obs, N] -> output [B, C, T_pred, N]. The first conv extrapolates
    T_obs -> T_pred; the remaining ``n_txpcnn - 1`` convs plus the output conv refine at
    the prediction horizon, with residual connections as in the reference implementation.
    """

    def __init__(self, obs_len: int, pred_len: int, n_txpcnn: int = 5, dropout: float = 0.0):
        super().__init__()
        assert n_txpcnn >= 1
        self.convs = nn.ModuleList([nn.Conv2d(obs_len, pred_len, 3, padding=1)])
        self.prelus = nn.ModuleList([nn.PReLU()])
        for _ in range(1, n_txpcnn):
            self.convs.append(nn.Conv2d(pred_len, pred_len, 3, padding=1))
            self.prelus.append(nn.PReLU())
        self.output = nn.Conv2d(pred_len, pred_len, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)                    # [B, T_obs, C, N]
        x = self.prelus[0](self.convs[0](x))
        for conv, prelu in zip(self.convs[1:], self.prelus[1:]):
            x = self.dropout(prelu(conv(x))) + x
        x = self.output(x)
        return x.permute(0, 2, 1, 3)                 # [B, C, T_pred, N]
