import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as ts

from torch_geometric.nn import GATConv
from models.nereus.params import NEREUSParams

import torch_scatter as ts

class GAT(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.pre_gnn_norm = nn.LayerNorm(config.enc_hidden_size)
        self.gnn = GATConv(
            in_channels=config.enc_hidden_size,
            out_channels=config.gnn_hidden_size,
            heads=config.gnn_n_head,
            edge_dim=config.edge_feat_dim,
            concat=False,
        )
        self.dropout_layer = nn.Dropout(0.1)

    def forward(self, h_enc, edge_index, edge_attr):

        h_enc_norm = self.pre_gnn_norm(h_enc)
        h_gnn = self.gnn(h_enc_norm, edge_index, edge_attr)
        h_gnn = self.dropout_layer(h_gnn)

        return h_gnn


class SocialPoolFast(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.num_rings = 6
        self.num_wedges = 6

        self.register_buffer("rmin", torch.tensor(5.0))
        self.register_buffer("rmax", torch.tensor(500.0))
        self.register_buffer("two_pi", torch.tensor(2.0 * torch.pi))

    def forward(self, y_pred, hidden, ego_idx, edge_index):
        """
        y_pred:  [N, 1, 2]
        hidden:  [N, 1, H]
        ego_idx: [E]
        edge_index: [2, E]
        """

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

            ego_pos = y_pred[ego, 0]         # [2]
            nbr_pos = y_pred[nbrs, 0]        # [K, 2]
            nbr_hid = hidden[nbrs, 0]        # [K, H]

            # relative positions
            diff = nbr_pos - ego_pos         # [K, 2]
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
            ring = torch.floor(self.num_rings * r_normed).long().clamp(0, self.num_rings - 1)

            theta = torch.atan2(diff[:, 1], diff[:, 0])
            theta_normed = torch.remainder(theta, self.two_pi) / self.two_pi
            wedge = torch.floor(self.num_wedges * theta_normed).long().clamp(0, self.num_wedges - 1)

            bin_id = ring * self.num_wedges + wedge   # [K]

            pooled = ts.scatter_mean(
                nbr_hid,
                bin_id,
                dim=0,
                dim_size=num_bins
            )

            pooled = pooled.mean(dim=0)   # mean over bins → [H]
            out[ego, 0] = pooled

        return out
    
    def bin_indices_vectorized(self, ydash):
        """
        ydash: [S, T, 2]
        returns: [S, S, T] in [0, num_bins-1]
        """
        with torch.no_grad():
            device = ydash.device
            S, T, _ = ydash.shape

            diff = ydash[:, None, :, :] - ydash[None, :, :, :]
            r = torch.norm(diff, dim=-1)

            mask_self = torch.eye(S, device=device).bool().unsqueeze(-1)
            mask_outside = r >= self.rmax

            r_safe = torch.clamp(r, min=self.rmin, max=self.rmax - 1e-6)

            r_normed = torch.log(r_safe / self.rmin) / torch.log(self.rmax / self.rmin)
            ring_ids = torch.floor(self.num_rings * r_normed).long()
            ring_ids = ring_ids.clamp(0, self.num_rings - 1)

            theta = torch.atan2(diff[..., 1], diff[..., 0])
            theta_normed = torch.remainder(theta, self.two_pi) / self.two_pi
            wedge_ids = torch.floor(self.num_wedges * theta_normed).long()
            wedge_ids = wedge_ids.clamp(0, self.num_wedges - 1)

            final_index = 1 + ring_ids * self.num_wedges + wedge_ids

            # 🔒 CRITICAL: zero invalid BEFORE returning
            final_index[mask_self | mask_outside] = 0

        return final_index

    

class EgoSocialPooling(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()

        self.social_pool = SocialPoolFast(config)
        self.pre_norm = nn.LayerNorm(config.enc_hidden_size)
        self.proj = nn.Linear(
            config.enc_hidden_size,
            config.gnn_hidden_size
        )

    def forward(self, h_enc, data):
        h_enc = self.pre_norm(h_enc)

        pooled = self.social_pool(
            y_pred=data.x_pos[:, -1:, :],        # [N, 1, 2]
            hidden=h_enc.unsqueeze(1),           # [N, 1, H]
            ego_idx=data.is_ego.nonzero(as_tuple=True)[0],
            edge_index=data.edge_index
        )                                        # [N, 1, H]

        pooled = pooled.squeeze(1)               # [N, H]
        pooled = self.proj(pooled)               # [N, gnn_hidden]

        return pooled

