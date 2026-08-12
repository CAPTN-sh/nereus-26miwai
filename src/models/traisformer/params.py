from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE


@dataclass
class TraisformerParams:
    pred_scope: str = "path" # ["path", "destination"]
    intent_head = "factorized" # "linear", "factorized" "cnn", "mixture", "lowrank"
    k_rank = 15

    max_dist: int = 0 # turn of edges in data loader
    pred_len: int = 0 * STEPS_PER_MINUTE
    obs_len: int =  10 * STEPS_PER_MINUTE

    # Model dims
    n_head: int = 4
    n_layer: int = 4

    # state_embd
    n_embd = 128

    n_chanels: int = 4
    n_vessel_feat: int = 8

    # Dropouts
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # loss
    coarse_loss_beta: float = 1.0
    coarse_loss_pool_size: int = 3
