import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as ts

from models.desire.params import DESIREParams

class SocialPool(nn.Module):
    """
    Social Pooling that bins the surrounding vessels depending on distance and angle.
    Bins are then pooled and a Linear layer summarizes the bins for final encoding.
    """
    def __init__(self, params: DESIREParams):
        super(SocialPool, self).__init__()
        self.hidden_size = params.hidden_size
        self.num_rings = params.num_rings
        self.num_wedges = params.num_wedges
        self.input_size = self.num_wedges * self.num_rings * self.hidden_size

        self.register_buffer("rmin",   torch.tensor(float(params.rmin)))
        self.register_buffer("rmax",   torch.tensor(float(params.max_dist)))
        self.register_buffer("two_pi", torch.tensor(2.0 * torch.pi))

        self.fc = nn.Linear(self.input_size, self.hidden_size)

    def forward(self, pred_pos_abs, hidden, data):
        device = pred_pos_abs.device
        N, K, T, H = hidden.shape
        num_bins = 1 + self.num_rings * self.num_wedges

        out = torch.zeros(N, K, self.hidden_size, device=device)
        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
        row, col = data.edge_index

        for ego in ego_idx:
            mask = (row == ego)
            nbrs = col[mask]

            if nbrs.numel() == 0:
                continue

            # ---- positions at t0 ----
            ego_pos = pred_pos_abs[ego, :, 0]
            nbr_pos = pred_pos_abs[nbrs, :, 0]
            nbr_hid = hidden[nbrs, :, 0] 

            # relative geometry per hypothesis
            diff = nbr_pos - ego_pos.unsqueeze(0)
            r = torch.norm(diff, dim=-1)

            mask_valid = r < self.rmax
            if not mask_valid.any():
                continue

            diff = diff[mask_valid]
            r = r[mask_valid]
            nbr_hid = nbr_hid[mask_valid]

            # ---- binning ----
            r_safe = torch.clamp(r, min=self.rmin, max=self.rmax - 1e-6)
            r_normed = torch.log(r_safe / self.rmin) / torch.log(self.rmax / self.rmin)
            ring = torch.floor(self.num_rings * r_normed).long().clamp(0, self.num_rings - 1)

            theta = torch.atan2(diff[..., 1], diff[..., 0])
            theta_normed = torch.remainder(theta, self.two_pi) / self.two_pi
            wedge = torch.floor(self.num_wedges * theta_normed).long().clamp(0, self.num_wedges - 1)

            bin_id = ring * self.num_wedges + wedge  # [valid_pairs]

            pooled = ts.scatter_mean(
                nbr_hid,
                bin_id,
                dim=0,
                dim_size=num_bins
            )

            pooled = pooled.reshape(-1)
            pooled = self.fc(pooled)

            out[ego] = pooled

        # expand over T (since GRU expects [N,K,T,H])
        out = out.unsqueeze(2).expand(-1, -1, T, -1)

        return F.relu(out)

    def bin_indices(self, ydash):
        """
        ydash: [N, T, 2]
        returns: [N, N, T] bin indices
        """
        with torch.no_grad():
            device = ydash.device
            N, T, _ = ydash.shape

            diff = ydash[:, None, :, :] - ydash[None, :, :, :]
            r = torch.norm(diff, dim=-1)

            # ring indices
            r_safe = torch.clamp(r, min=self.rmin)
            r_normed = torch.log(r_safe / self.rmin) / torch.log(self.rmax / self.rmin)
            ring_ids = torch.ceil(self.num_rings * r_normed).long()
            ring_ids = ring_ids.clamp(0, self.num_rings - 1)

            # angle
            theta = torch.atan2(diff[..., 1], diff[..., 0])      # [N, N, T]
            theta_normed = torch.remainder(theta, self.two_pi) / self.two_pi
            wedge_ids = torch.floor(self.num_wedges * theta_normed).long()

            # final_index
            final_index = 1 + ring_ids * self.num_wedges + wedge_ids
            mask_self = torch.eye(N, device=device).bool().unsqueeze(-1)
            mask_outside = r >= self.rmax
            final_index[mask_self | mask_outside] = 0

        return final_index
