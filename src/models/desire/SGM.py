import torch
import torch.nn as nn

from models.desire.nn.cvae import CVAEEncoder
from models.desire.nn.rnn import GRUDecoder
from models.utils.rnn import GRUEncoder
from models.desire.utils.params import DESIREParams


class SGM(nn.Module):
    """
    Sample Generation Module (DESIRE) — K-sample

    3.1. Diverse Sample Generation with CVAE
    """

    def __init__(self, params: DESIREParams):
        super().__init__()
        self.pred_len = params.pred_len
        self.pred_dim = params.pred_dim

        self.hidden_size = params.hidden_size
        self.latent_size = params.latent_size
        self.num_samples = params.num_samples

        self.enc_obs = GRUEncoder(params.hidden_size, params.kin_dim)
        self.enc_fut = GRUEncoder(params.hidden_size, params.pred_dim)

        self.cvae = CVAEEncoder(params)
        self.beta_fc = nn.Linear(self.latent_size, self.hidden_size, bias=True)

        self.dec = GRUDecoder(params) #TODO use utils

    def forward(self, data):
        device = data.x.device
        N = data.x.shape[0]
        K = self.num_samples

        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]

        hidden_obs = self.enc_obs(data.x, data.x_mask)
        hidden_fut = self.enc_fut(data.y_rel_pos, data.y_mask)

        mean_ego, log_var_ego = self.cvae(hidden_fut, hidden_obs[ego_idx])

        std = (0.5 * log_var_ego).exp()
        eps = torch.randn(mean_ego.size(0), K, self.latent_size, device=device)

        z_ego = mean_ego.unsqueeze(1) + std.unsqueeze(1) * eps
        z_all = torch.randn(N, K, self.latent_size, device=device)
        z_all[ego_idx] = z_ego

        pred_pos_rel = self.generate_traj(hidden_obs, z_all)

        return pred_pos_rel, hidden_obs, mean_ego, log_var_ego
    
    def inference(self, data):

        device = data.x.device
        N = data.x.shape[0]
        K = self.num_samples

        hidden_obs = self.enc_obs(data.x, data.x_mask)

        z_all = torch.randn(N, K, self.latent_size, device=device)

        pred_pos_rel = self.generate_traj(hidden_obs, z_all)

        mean = torch.zeros(K, self.latent_size, device=device)
        log_var = torch.zeros(K, self.latent_size, device=device)

        return pred_pos_rel, hidden_obs, mean, log_var
    
    def generate_traj(self, hidden_all, z):
        N, K, _ = z.shape
        H = self.hidden_size

        # guided dropout
        beta = torch.softmax(self.beta_fc(z), dim=-1)

        # TODO condition on static features
        x_t0 = hidden_all.unsqueeze(1) * beta
        x_t0 = x_t0.view(1, N * K, H)

        dec_in = torch.zeros(N * K, self.pred_len, H, device=x_t0.device)

        pred_pos_rel, _ = self.dec(dec_in, x_t0)
        pred_pos_rel = pred_pos_rel.view(N, K, self.pred_len, self.pred_dim)

        return pred_pos_rel