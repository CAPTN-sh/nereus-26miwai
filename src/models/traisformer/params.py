from dataclasses import dataclass


@dataclass
class TraisformerParams:
    # Sequence lengths
    # TODO from data loader config
    obs_len: int = 24
    pred_len: int = 36
    max_seqlen = obs_len + pred_len

    # Model dims
    n_head: int = 8
    n_layer: int = 8
    n_x_embd: int = 128
    n_y_embd: int = 128
    n_embd: int = 256 + 128  # must equal n_embd_x + n_embd_y

    # Dropouts
    pdrop: float = 0.1

    # Intent head specifics
    intent_from: str = "obs_last"  # "obs_last" or "obs_mean": which hidden to use
    intent_gauss_sigma_px: float = 1.2  # ~1-2 px blur as requested
    intent_target_eps: float = 1e-8  # to avoid log(0) when normalizing
