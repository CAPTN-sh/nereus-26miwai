from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE

@dataclass
class TraisformerParams:
    # Sequence lengths
    pred_scope: str = "path" # ["path", "destination"]
    intent_head = "factorized" # "linear", "factorized" "cnn", "mixture", "lowrank"
    k_rank = 15
    
    pred_len: int = 0 * STEPS_PER_MINUTE
    obs_len: int =  10 * STEPS_PER_MINUTE

    # bbox
    # TODO from config
    bbox = [10.12, 54.31, 10.33, 54.46]

    # Model dims
    n_head: int = 4
    n_layer: int = 3

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