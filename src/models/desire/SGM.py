import torch
import torch.nn as nn

from models.desire.nn.cvae import CVAEEncoder
from models.desire.nn.rnn import RNNDecoder, RNNEncoder
from models.desire.utils.params import DESIREParams


class SGM(nn.Module):
    """
    Sample Generation Module (DESIRE) — K-sample

    3.1. Diverse Sample Generation with CVAE
    """

    def __init__(self, params: DESIREParams):
        super().__init__()
        self.pred_len = params.pred_len
        self.hidden_size = params.hidden_size
        self.latent_size = params.latent_size
        self.num_samples = params.num_samples

        self.enc_obs = RNNEncoder(params, kernel_size=3)
        self.enc_fut = RNNEncoder(params, kernel_size=1)

        self.cvae = CVAEEncoder(params)
        self.beta_fc = nn.Linear(self.latent_size, self.hidden_size, bias=True)

        self.dec = RNNDecoder(params)

    def forward(self, obs_pos_rel: torch.Tensor, fut_pos_rel: torch.Tensor):
        device = obs_pos_rel.device
        B = fut_pos_rel.shape[0]

        # encode obs and fut
        hidde_obs_enc = self.enc_obs(obs_pos_rel)[1][-1]
        hidde_fut_enc = self.enc_fut(fut_pos_rel)[1][-1]

        # sample k times
        mean, log_var = self.cvae(hidde_fut_enc, hidde_obs_enc)
        std = (0.5 * log_var).exp().unsqueeze(1)
        eps = torch.randn(B, self.num_samples, self.latent_size, device=device)
        z_k = mean.unsqueeze(1) + std * eps

        pred_pos_rel = self.generate_traj(hidde_obs_enc, z_k, B, device)
        return pred_pos_rel, hidde_obs_enc, mean, log_var

    def inference(self, obs_pos_rel: torch.Tensor):
        device = obs_pos_rel.device
        B = obs_pos_rel.size(0)
        K = self.num_samples
        L = self.latent_size

        # Encode observed
        hidde_obs_enc = self.enc_obs(obs_pos_rel)[1][-1]

        # Sample z_k ~ N(0,I) for prior
        z_k = torch.randn(B, K, L, device=device)
        mean = torch.zeros(B, K, L, device=device)
        log_var = torch.zeros(B, K, L, device=device)

        pred_pos_rel = self.generate_traj(hidde_obs_enc, z_k, B, device)
        return pred_pos_rel, hidde_obs_enc, mean, log_var

    def generate_traj(self, hidde_obs_enc, z_k, B, device):
        # guided drop out
        beta = torch.softmax(self.beta_fc(z_k), dim=-1)
        x_t0 = hidde_obs_enc.unsqueeze(1) * beta

        # Decoding
        zeros_tail = torch.zeros(
            B * self.num_samples, self.pred_len - 1, self.hidden_size, device=device
        )
        x_t0 = x_t0.view(B * self.num_samples, 1, self.hidden_size).contiguous()
        x_seq = torch.cat([x_t0, zeros_tail], dim=1)

        pred_pos_rel, _ = self.dec(x_seq)
        pred_pos_rel = pred_pos_rel.view(
            B, self.num_samples, 2, self.pred_len
        ).contiguous()

        return pred_pos_rel
