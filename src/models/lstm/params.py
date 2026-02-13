from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE

@dataclass
class LSTMParams:
    # Sequence lengths
    pred_len: int = 5 * STEPS_PER_MINUTE
    obs_len: int =  10 * STEPS_PER_MINUTE

    # Model dims
    # https://ieeexplore.ieee.org/document/9054421
    rnn_hidden_size: int = 128
    rnn_hidden_size: int = 128
