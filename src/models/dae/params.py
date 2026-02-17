from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE

@dataclass
class DAEParams:
    obs_len: int =  10 * STEPS_PER_MINUTE
    pred_len: int = 1 # not 0 otherwise heatmap is generated
    max_dist: int = 0

    n_traj_feat : int = 8
    n_vessel_feat : int = 8

    latent_dim: int = 32
    d_model : int = 128
    nhead : int = 4
    n_layer : int = 4
    dropout = 0.1

    # DAE-specific
    noise_std = 0.05