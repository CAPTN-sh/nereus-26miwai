from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE

@dataclass
class NEREUSParams:
    # Sequence lengths
    pred_len: int = 5 * STEPS_PER_MINUTE
    obs_len: int =  10 * STEPS_PER_MINUTE
    mdn_modes = 5

    node_feat_dim: int = 8
    edge_feat_dim: int = 25
    static_feat_dim: int = 8

    rnn_hidden_size: int = 384

    gnn_hidden_size: int = 128
    gnn_n_head: int = 4
    gnn_max_dist: int = 500

    # bbox
    # TODO from config
    bbox = [10.12, 54.31, 10.33, 54.46]
    map_cnn_out: int = 128
    prior_cnn_out: int = 128

    prior_pred_scope: str = "path" # ["path", "destination"] (TrAISformer)