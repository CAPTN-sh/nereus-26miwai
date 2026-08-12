from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE


@dataclass
class NEREUSParams:
    # Sequence lengths
    pred_len: int = 5 * STEPS_PER_MINUTE
    obs_len: int =  10 * STEPS_PER_MINUTE
    node_feat_dim: int = 8
    static_feat_dim: int = 8
    edge_feat_dim: int = 25

    mdn_modes: int = 3

    rnn_hidden_size: int = 256

    gnn_hidden_size: int = 64
    gnn_n_head: int = 4
    max_dist: int = 500

    map_cnn_in: int = 4
    map_cnn_out: int = 128
    map_radius: int = 500
    map_res: int = 50

    prior_cnn_out: int = 128

    prior_pred_scope: str = "path" # ["path", "destination"] (TrAISformer)
