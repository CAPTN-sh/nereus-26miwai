import torch
from torch import nn
from torch_geometric.data import Data

from models.isstgcnn.modules.graph import build_adjacency, densify, normalize_adj
from models.isstgcnn.modules.mpc import mpc_correct
from models.isstgcnn.modules.stgcnn import STGCNBlock, TXPCNN
from models.isstgcnn.params import ISSTGCNNParams

POS_SCALE = 100.0


class ISSTGCNN(nn.Module):
    """IS-STGCNN (Feng et al., Ocean Engineering 266, 2022).

    Social-STGCNN's ST-GCNN encoder and TXP-CNN decoder over a spatio-temporal graph,
    with the paper's three additions available independently through the config:
    the Mahalanobis adjacency kernel (``adj_kernel``), social sampling (handled in
    ``loss.py``), and the MPC trajectory correction (``use_mpc``). Setting
    ``adj_kernel="euclid"``, ``social_sampling=False`` and ``use_mpc=False`` recovers
    plain Social-STGCNN, which is the paper's own baseline.

    Every node in the scene gets a prediction; ``inference`` returns the ego's, in the
    ``(best, K samples)`` form the eval pipeline expects.
    """

    def __init__(self, config: ISSTGCNNParams):
        super().__init__()
        self.cfg = config

        hidden = config.stgcnn_hidden
        blocks = [STGCNBlock(2, hidden, config.kernel_size, config.dropout)]
        blocks += [
            STGCNBlock(hidden, hidden, config.kernel_size, config.dropout)
            for _ in range(config.n_stgcnn - 1)
        ]
        self.st_gcns = nn.ModuleList(blocks)
        self.txp_cnn = TXPCNN(config.obs_len, config.pred_len, config.n_txpcnn, config.dropout)
        self.proj = (
            nn.Identity() if hidden == config.out_dim
            else nn.Conv2d(hidden, config.out_dim, kernel_size=1)
        )

    def forward(self, data: Data, scene=None):
        """Returns ``(params [B, N, T_pred, 5], dense)``.

        ``dense`` is the densified batch (see ``modules.graph.densify``); the loss needs
        it for the all-node targets and for placing social-sampling negatives.
        """
        dense = densify(data, self.cfg.max_neighbors + 1)

        a = build_adjacency(dense["pos"], dense["node_mask"], dense["step_mask"], self.cfg.adj_kernel)
        a = normalize_adj(a, dense["node_mask"])

        x = dense["v"].permute(0, 3, 2, 1)            # [B, 2, T_obs, N]
        for block in self.st_gcns:
            x = block(x, a)
        x = self.proj(self.txp_cnn(x))                # [B, 5, T_pred, N]

        return x.permute(0, 3, 2, 1), dense           # [B, N, T_pred, 5]

    def inference(self, data: Data, scene=None):
        """Returns ``(best_rel [B, T, 2], k_rel [B, K, T, 2])`` for the ego vessels.

        ``best_rel`` is the mean of the predicted bivariate Gaussian -- the trajectory
        the paper's ADE/FDE are computed on -- optionally passed through the MPC
        corrector. ``k_rel`` are ``num_samples`` draws from the same Gaussian, used only
        for this repo's ``k_ade``/``k_fde`` columns; they are left uncorrected, since MPC
        is an inference-time smoother the paper applies to its single output.
        """
        params, dense = self.forward(data, scene)
        ego = dense["is_ego"]
        p = params[ego]                                # [B, T, 5]

        mu = p[..., 0:2]
        sigma = torch.exp(p[..., 2:4].clamp(-7.0, 5.0))
        rho = torch.tanh(p[..., 4]).clamp(-0.99, 0.99)

        z = torch.randn(p.size(0), self.cfg.num_samples, *mu.shape[1:], device=p.device)
        sx, sy = sigma[..., 0].unsqueeze(1), sigma[..., 1].unsqueeze(1)
        rho_k = rho.unsqueeze(1)
        dx = sx * z[..., 0]
        dy = sy * (rho_k * z[..., 0] + torch.sqrt((1.0 - rho_k ** 2).clamp_min(1e-6)) * z[..., 1])
        k_rel = mu.unsqueeze(1) + torch.stack([dx, dy], dim=-1)

        best_rel = mu
        if self.cfg.use_mpc:
            last_pos = dense["pos"][ego][:, -1]        # [B, 2]
            raw = dense["x_raw"][ego][:, -1]           # [B, 4] speed, course, acc, ang_diff
            pred_abs = torch.cumsum(best_rel, dim=1) * POS_SCALE + last_pos.unsqueeze(1)
            corrected = mpc_correct(pred_abs, last_pos, raw[:, 1], raw[:, 3], self.cfg)
            prev = torch.cat([last_pos.unsqueeze(1), corrected[:, :-1]], dim=1)
            best_rel = (corrected - prev) / POS_SCALE

        return best_rel, k_rel
