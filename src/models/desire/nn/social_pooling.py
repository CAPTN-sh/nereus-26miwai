import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as ts

from models.desire.utils.params import DESIREParams


class SocialPool(nn.Module):

    def __init__(self, params: DESIREParams):
        super(SocialPool, self).__init__()
        self.hidden_size = params.hidden_size
        self.num_rings = params.num_rings
        self.num_wedges = params.num_wedges
        self.input_size = self.num_wedges * self.num_rings * self.hidden_size

        self.register_buffer("rmin",   torch.tensor(float(params.rmin)))
        self.register_buffer("rmax",   torch.tensor(float(params.rmax)))
        self.register_buffer("two_pi", torch.tensor(2.0 * torch.pi))

        self.fc = nn.Linear(self.input_size, self.hidden_size)

    def forward(self, y_pred, hidden, seq_start_end):
        device = y_pred.device
        batch_size, _ = y_pred.size()
        out = torch.zeros(batch_size, self.input_size, device=device)
        num_bins = 1 + self.num_rings * self.num_wedges

        for (start, end) in seq_start_end:
            seq_size = end-start

            bin_ids = self.bin_indices(y_pred[start:end])
            agent_offset = torch.arange(seq_size, device=device).view(seq_size, 1) * num_bins
            global_idx = (bin_ids + agent_offset).reshape(-1)

            hidden_seq = hidden[start:end].repeat(seq_size, 1)
            pooled = ts.scatter_mean(hidden_seq, global_idx, dim=0, dim_size= seq_size * num_bins)
            
            without_dummy = pooled.view(seq_size, num_bins, self.hidden_size)[:, 1:, :]
            out[start:end, :] = without_dummy.reshape(seq_size, self.input_size)

        return F.relu(self.fc(out))
    

    def bin_indices(self, ydash):
        """Compute (ring, wedge) bin indices for all agent pairs in one scene."""

        with torch.no_grad():
            r = torch.norm(ydash[:, None] - ydash, dim=2, p=2)

            r_normed = (torch.log(r/self.rmin) / torch.log(self.rmax/self.rmin))
            ring_ids = torch.ceil(self.num_rings * r_normed).long().clamp(0, self.num_rings - 1)

            x_dist = (ydash[:, 0] - ydash[:, 0, None])
            y_dist = (ydash[:, 1] - ydash[:, 1, None])
            theta = torch.atan2(y_dist, x_dist)
            theta_normed = torch.remainder(theta, self.two_pi) / (self.two_pi)
            wedge_ids = torch.floor(self.num_wedges * theta_normed).long()
            
            mask_self = torch.eye(r.size(0), dtype=torch.bool, device=r.device)
            mask_outside = (r >= self.rmax)

            final_index = 1 + ring_ids * self.num_wedges + wedge_ids
            final_index[mask_self | mask_outside] = 0

        return final_index