from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE

@dataclass
class GNNLSTMParams:
    # Sequence lengths
    pred_len: int = 5 * STEPS_PER_MINUTE
    obs_len: int =  10 * STEPS_PER_MINUTE
    node_feat_dim: int = 15
    edge_feat_dim: int = 9


    enc_hidden_size: int = 256
    enc_n_layers: int = 1
    dec_hidden_size: int = 256

    gnn_hidden_size: int = 256
    gnn_n_head: int = 4