from torch import nn

from data.map.rasterize import Rasterizer
from models.desire.ioc_modules.scf import SCF
from models.desire.params import DESIREParams


class IOC(nn.Module):
    """Inverse optimal control (IOC) module
    - adds position dependent scene features from CNN and social pooling via SCF
    - per-step scoring RNN over SCF features
    - accumulated score per hypothesis
    - single refinement Δ
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

    def forward(self, pred_pos_rel, hidden_obs, data, scene_feats):
        N, K, T, C = pred_pos_rel.shape
        H = hidden_obs.size(-1)

        obs_pos_last = data.x_pos[:, -1].unsqueeze(1).expand(-1, K, -1)
        pred_pos_abs = obs_pos_last.unsqueeze(2) + pred_pos_rel.cumsum(dim=2) * 100.0

        h0 = hidden_obs.unsqueeze(1).expand(-1, K, -1)
        h0_proj = h0.unsqueeze(2).expand(-1, -1, T, -1)
        scf_seq = self.scf.forward(h0_proj, pred_pos_abs, pred_pos_rel, data, scene_feats)
        scf_seq = scf_seq.reshape(N*K, T, -1)

        h_seq, _ = self.gru(scf_seq, h0.reshape(N*K, H).unsqueeze(0))
        h_last = h_seq[:, -1]

        acc_scores = self.score_fc(h_seq).squeeze(-1).sum(dim=1).view(N, K)
        pred_delta = self.delta_fc(h_last).view(N, K, T, C)

        return acc_scores, pred_delta
