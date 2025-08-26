from dataclasses import dataclass

import torch.nn as nn


@dataclass
class DESIREParams:
    # general
    pred_dim: int = 2
    pred_len: int = 12
    num_refine_iters: int = 2

    hidden_size: int = 48
    intermediate_size: int = 16

    # scene_cnn
    in_channels: int = 4
    out_channels: int = 32

    # sgm
    latent_size: int = 48
    num_samples: int = 50

    # social pooling
    num_rings: int = 6
    num_wedges: int = 6
    rmin: int = 1
    rmax: int = 1000
