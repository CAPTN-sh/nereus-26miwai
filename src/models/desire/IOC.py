import torch
import torch.nn as nn

from models.desire.nn.scf import SCF
from models.desire.utils.params import DESIREParams
from models.utils.maps.rasterize import Rasterizer


class IOC(nn.Module):
    """
    K-hypothesis IOC:
      - scene features from CNN (HWC) + social pooling via SCF
      - per-step scoring RNN over SCF features
      - accumulated score per hypothesis [B,K]
      - single refinement Δ from last hidden [B,K,2,T]
    """
    def __init__(self, params: DESIREParams, rasterizer: Rasterizer):
        super().__init__()
        _h = params.hidden_size
        _out = params.pred_dim * params.pred_len
        _scf_out = params.hidden_size + params.intermediate_size + params.out_channels

        self.scf = SCF(params, rasterizer)
        self.gru = nn.GRU(input_size=_scf_out, hidden_size=_h, batch_first=True)
        self.score_fc = nn.Linear(_h, 1)
        self.delta_fc = nn.Linear(_h, _out)

    def forward(self, pred_pos_rel, hidde_obs_enc, obs_pos_last, seq_start_end, scene_feats):
        # pred_traj_rel: [B,K,2,T]; prev_hidden: [B,Hx]; obs_pos_t0/obs_last: [B,2]
        device = pred_pos_rel.device
        B, K, C, T = pred_pos_rel.shape

        # expand over the K samples
        obs_pos_last = obs_pos_last[:, None, :].expand(-1, K, -1).reshape(B * K, C).unsqueeze(-1)
        seq_start_end = (seq_start_end * K).to(device).long()
        h0 = hidde_obs_enc[:, None, :].expand(-1, K, -1).reshape(B * K, -1)
        
        # build absolute coords
        pred_pos_rel = pred_pos_rel.reshape(B * K, C, T)
        pred_pos_abs = obs_pos_last + pred_pos_rel.cumsum(dim=-1) * 100.0
        
        h0_proj = h0.unsqueeze(1).expand(-1, T, -1)
        scf_seq = self.scf.forward_vectorized(h0_proj, pred_pos_abs, pred_pos_rel, seq_start_end, scene_feats)
        h_seq, _ = self.gru(scf_seq, h0.unsqueeze(0)) 
        h_last = h_seq[:, -1]
        
        acc_scores = self.score_fc(h_seq).squeeze(-1).sum(dim=1)
        acc_scores = acc_scores.view(B, K)
        pred_delta = self.delta_fc(h_last).view(B, K, C, T)

        return acc_scores, pred_delta