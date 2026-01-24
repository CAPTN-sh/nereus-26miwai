from dataclasses import dataclass

from utils.config import STEPS_PER_MINUTE

@dataclass
class LSTMParams:
    # Sequence lengths
    # TODO from data loader config
    pred_len: int = 5 * STEPS_PER_MINUTE
    obs_len: int =  10 * STEPS_PER_MINUTE

    # Model dims
    # https://ieeexplore.ieee.org/document/9054421
    enc_hidden_size: int = 128
    dec_hidden_size: int = 128
