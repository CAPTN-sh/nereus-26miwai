from dataclasses import dataclass


@dataclass
class LSTMParams:
    # Sequence lengths
    # TODO from data loader config
    pred_len: int = 3 * 12 # 3 min
    obs_len: int =  8 * 12 # 8 min

    # Model dims
    # https://ieeexplore.ieee.org/document/9054421
    enc_hidden_size: int = 64
    dec_hidden_size: int = 64
