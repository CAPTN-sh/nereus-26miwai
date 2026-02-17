import torch
from torch import nn

from models.dae.params import DAEParams
from models.dae.encoder import TransformerEncoder
from models.dae.decoder import TransformerDecoder

class DAE(nn.Module):
    def __init__(self, cfg: DAEParams):
        super().__init__()
        self.noise_std = cfg.noise_std
        self.encoder = TransformerEncoder(cfg)
        self.decoder = TransformerDecoder(cfg)

    def forward(self, data, scene = None):
        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
        traj_feat = data.x[ego_idx, :, :]
        static_feat = data.static[ego_idx, :]
        mask = data.x_mask[ego_idx, :]

        noise = torch.randn_like(traj_feat) * self.noise_std
        traj_feat = traj_feat + noise * mask.unsqueeze(-1)

        latent = self.encoder(traj_feat, static_feat, mask)
        rec = self.decoder(latent, mask)

        return rec, latent
    
    def inference(self, data, scene = None):
        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
        traj_feat = data.x[ego_idx, :, :]
        static_feat = data.static[ego_idx, :]
        mask = data.x_mask[ego_idx, :]

        latent = self.encoder(traj_feat, static_feat, mask)
        rec = self.decoder(latent, mask)

        return rec, latent
    
    @torch.no_grad()
    def _make_noisy(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.noise_std