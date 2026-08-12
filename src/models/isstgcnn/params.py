from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE


@dataclass
class ISSTGCNNParams:
    """Hyperparameters for IS-STGCNN (Feng et al., Ocean Engineering 266, 2022).

    Defaults follow the paper where it states a value (1 ST-GCNN layer, 5 TXP-CNN
    layers, 1:3 positive:negative sampling ratio, |delta| <= 35 deg) and this repo's
    conventions otherwise (sequence lengths, the /100 displacement normalisation).
    """

    # Sequence lengths (obs_len / pred_len / max_dist are also read by AISDataModule)
    pred_len: int = 5 * STEPS_PER_MINUTE
    obs_len: int = 10 * STEPS_PER_MINUTE
    max_dist: int = 500

    # ST-GCNN / TXP-CNN stack
    n_stgcnn: int = 1        # paper: best performance with 1 layer
    n_txpcnn: int = 5        # paper: best performance with 5 layers
    # Social-STGCNN carries out_dim channels all the way through (a ~7k parameter
    # model). A wider hidden size adds a final 1x1 projection to out_dim.
    stgcnn_hidden: int = 5
    out_dim: int = 5         # bivariate Gaussian: mu_x, mu_y, log_sx, log_sy, atanh(rho)
    kernel_size: int = 3     # temporal conv kernel of the ST-GCNN block
    dropout: float = 0.0
    max_neighbors: int = 16  # cap on neighbours per graph; bounds the dense [B,T,N,N] kernel

    # Graph construction
    adj_kernel: str = "mahalanobis"  # "euclid" | "mahalanobis" | "eq4_literal"

    # Supervision / sampling
    supervise: str = "all"   # "all" (paper) | "ego" (this repo's other baselines)
    num_samples: int = 3     # K for the k_ade / k_fde columns (matches NEREUS's mdn_modes)

    # Social-sampling (paper 3.3)
    social_sampling: bool = True
    negative_mode: str = "bumper"  # "bumper" (paper V3) | "random" (paper V2 baseline)
    random_radius: float = 500.0   # metres, for negative_mode="random"
    n_negatives: int = 3          # paper: positive:negative = 1:3
    lambda_social: float = 0.5
    bumper_b: float = 6.4         # forward semi-major axis, in ship lengths (Hara 1991)
    bumper_a: float = 1.6         # abeam / aft semi-minor axis, in ship lengths
    collision_frac: float = 0.25  # inner "collision" band as a fraction of the bumper axes
    collision_weight: float = 1.0  # 1.0 => collision and risk negatives weighted alike

    # MPC trajectory correction (paper 3.4) -- inference only
    use_mpc: bool = False
    nomoto_k: float = 0.1      # Nomoto gain K [1/s]
    nomoto_t: float = 30.0     # Nomoto time constant T [s]
    rudder_max_deg: float = 35.0
    mpc_iters: int = 50
    mpc_lr: float = 0.1
