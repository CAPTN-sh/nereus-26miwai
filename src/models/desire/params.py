from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE


@dataclass
class DESIREParams:
    # general (fix)
    pred_dim: int = 2
    obs_len: int = 10 * STEPS_PER_MINUTE
    pred_len: int = 5 * STEPS_PER_MINUTE

    num_refine_iters: int = 2
    hidden_size: int = 256
    intermediate_size: int = 32

    # scene_cnn
    in_channels: int = 4
    out_channels: int = 32

    # sgm
    kin_dim = 8
    static_feat_dim = 8
    latent_size: int = 32
    num_samples: int = 2

    # social pooling
    num_rings: int = 6
    num_wedges: int = 6
    rmin: int = 5
    max_dist: int = 1000
