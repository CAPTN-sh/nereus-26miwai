from dataclasses import dataclass

import torch.nn as nn


@dataclass
class DESIREParams:
    # general
    pred_dim: int = 2
    pred_len: int = 36
    num_refine_iters: int = 1

    hidden_size: int = 32
    intermediate_size: int = 16

    # scene_cnn
    in_channels: int = 2
    out_channels: int = 16

    # sgm
    latent_size: int = 32
    num_samples: int = 6

    # social pooling
    num_rings: int = 6
    num_wedges: int = 6
    rmin: int = 5
    rmax: int = 1000
