from dataclasses import dataclass


@dataclass
class TraisformerParams:
    # Sequence lengths
    pred_scope: str = "destination" # ["path", "destination"]
    intent_head = "linear" # "linear", "factorized" "cnn", "mixture", "lowrank"
    k_rank = 16
    
    pred_len: int = 1 * 12
    obs_len: int = 10 * 12 # fix at 10min

    # bbox
    # TODO from config
    bbox = [10.12, 54.31, 10.33, 54.46]

    # Model dims
    n_head: int = 8
    n_layer: int = 8

    # state_embd
    n_spatial_embd: int = 128
    n_kinematic_embd: int = 64
    n_dynamic_embd: int = 32

    n_chanels: int = 4
    n_terrain_embd: int = 64

    n_vessel_feat: int = 7
    n_vessel_embd: int = 32

    n_embd = 2*128 + 2*64 + 2*32

    # Dropouts
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # loss
    coarse_loss_beta: float = 1.0
    coarse_loss_pool_size: int = 3