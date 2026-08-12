import torch
import torch_scatter as ts
from torch import nn

from models.nereus.params import NEREUSParams


class SocialPoolFast(nn.Module):
    """Social pooling through bins depending on relative position and angle.
    """

    def __init__(self, config: NEREUSParams):
        super().__init__()

        self.num_rings = 6
        self.num_wedges = 6

        self.register_buffer("rmin", torch.tensor(5.0))
        self.register_buffer("rmax", torch.tensor(config.max_dist))
        self.register_buffer("two_pi", torch.tensor(2.0 * torch.pi))

    def forward(self, y_pred, hidden, ego_idx, edge_index):
        device = y_pred.device
        N, _, H = hidden.shape
        out = torch.zeros(N, 1, H, device=device)

        # build neighbor lists (once)
        neighbors = [[] for _ in range(N)]
        for i, j in edge_index.t().tolist():
            neighbors[i].append(j)
            neighbors[j].append(i)

        num_bins = self.num_rings * self.num_wedges

        for ego in ego_idx.tolist():
            nbrs = neighbors[ego]
            if len(nbrs) == 0:
                continue

            nbrs = torch.tensor(nbrs, device=device)

            ego_pos = y_pred[ego, 0]  # [2]
            nbr_pos = y_pred[nbrs, 0]  # [K, 2]
            nbr_hid = hidden[nbrs, 0]  # [K, H]

            # relative positions
            diff = nbr_pos - ego_pos  # [K, 2]
            r = torch.norm(diff, dim=-1)

            mask = r < self.rmax
            diff = diff[mask]
            r = r[mask]
            nbr_hid = nbr_hid[mask]

            if diff.numel() == 0:
                continue

            # ---- binning ----
            r_safe = torch.clamp(r, min=self.rmin, max=self.rmax - 1e-6)
            r_normed = torch.log(r_safe / self.rmin) / torch.log(self.rmax / self.rmin)
            ring = (
                torch
                .floor(self.num_rings * r_normed)
                .long()
                .clamp(0, self.num_rings - 1)
            )

            theta = torch.atan2(diff[:, 1], diff[:, 0])
            theta_normed = torch.remainder(theta, self.two_pi) / self.two_pi
            wedge = (
                torch
                .floor(self.num_wedges * theta_normed)
                .long()
                .clamp(0, self.num_wedges - 1)
            )

            bin_id = ring * self.num_wedges + wedge  # [K]

            pooled = ts.scatter_mean(nbr_hid, bin_id, dim=0, dim_size=num_bins)

            pooled = pooled.mean(dim=0)  # mean over bins → [H]
            out[ego, 0] = pooled

        return out
