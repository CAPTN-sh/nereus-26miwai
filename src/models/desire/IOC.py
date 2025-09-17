import torch
import torch.nn as nn

from models.desire.nn.scf import SCF
from models.desire.utils.params import DESIREParams


class IOC(nn.Module):
    """
    K-hypothesis IOC:
      - scene features from CNN (HWC) + social pooling via SCF
      - per-step scoring RNN over SCF features
      - accumulated score per hypothesis [B,K]
      - single refinement Δ from last hidden [B,K,2,T]
    """
    def __init__(self, params: DESIREParams):
        super().__init__()
        _h = params.hidden_size
        _out = params.pred_dim * params.pred_len
        _scf_out = params.hidden_size + params.intermediate_size + params.out_channels

        self.scf = SCF(params)
        self.gru_cell = nn.GRUCell(_scf_out, _h)
        self.score_fc = nn.Linear(_h, 1)
        self.delta_fc = nn.Linear(_h, _out)

    def forward(self, pred_pos_rel, hidde_obs_enc, obs_pos_last, seq_start_end, scene_feats, scene_meta):
        # pred_traj_rel: [B,K,2,T]; prev_hidden: [B,Hx]; obs_pos_t0/obs_last: [B,2]
        device = pred_pos_rel.device
        B, K, C, T = pred_pos_rel.shape

        # expand over the K samples
        obs_pos_last = obs_pos_last[:, None, :].expand(-1, K, -1).reshape(B * K, C).unsqueeze(-1)
        seq_start_end = (seq_start_end * K).to(device).long()
        h = hidde_obs_enc[:, None, :].expand(-1, K, -1).reshape(B * K, -1)
        
        # build absolute coords
        pred_pos_rel = pred_pos_rel.reshape(B * K, C, T)
        pred_pos_abs = obs_pos_last + pred_pos_rel.cumsum(dim=-1)
        
        acc_scores = torch.zeros(B*K, device=device)

        start = (T - 1) % 4 
        for t in range(start, T, 4):
            pa_t = pred_pos_abs[:, :, t]
            pr_t = pred_pos_rel[:, :, t]

            scf_t = self.scf(h, pa_t, pr_t, seq_start_end, scene_feats, scene_meta)
            h = self.gru_cell(scf_t, h)
            acc_scores = acc_scores + self.score_fc(h).squeeze(-1)

        acc_scores = acc_scores.view(B, K)
        pred_delta = self.delta_fc(h).view(B, K, C, T)
        return acc_scores, pred_delta