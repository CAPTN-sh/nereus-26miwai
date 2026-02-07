from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE

@dataclass
class TraisformerParams:
    # Sequence lengths
    pred_scope: str = "path" # ["path", "destination"]
    intent_head = "factorized" # "linear", "factorized" "cnn", "mixture", "lowrank"
    k_rank = 16
    
    pred_len: int = 20 * STEPS_PER_MINUTE
    obs_len: int =  10 * STEPS_PER_MINUTE

    # bbox
    # TODO from config
    bbox = [10.12, 54.31, 10.33, 54.46]

    # Model dims
    n_head: int = 8
    n_layer: int = 4

    # state_embd
    n_embd = 128
    n_spatial_embd: int = 32
    n_kinematic_embd: int = 16
    n_dynamic_embd: int = 16

    n_chanels: int = 4
    n_vessel_feat: int = 8

    # Dropouts
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # loss
    coarse_loss_beta: float = 1.0
    coarse_loss_pool_size: int = 3