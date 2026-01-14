from dataclasses import dataclass

import torch.nn as nn


@dataclass
class DESIREParams:
    # general (fix)
    pred_dim: int = 2
    obs_len: int = 12 * 12
    pred_len: int = 3 * 12

    num_refine_iters: int = 3
    hidden_size: int = 32
    intermediate_size: int = 16

    # scene_cnn
    in_channels: int = 4
    out_channels: int = 16

    # sgm
    obs_feat_dim = 11
    latent_size: int = 32
    num_samples: int = 6

    # social pooling
    num_rings: int = 6
    num_wedges: int = 6
    rmin: int = 5
    rmax: int = 2000