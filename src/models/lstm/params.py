from dataclasses import dataclass


@dataclass
class LSTMParams:
    # Sequence lengths
    # TODO from data loader config
    pred_len: int = 5 * 12
    obs_len: int =  10 * 12

    # Model dims
    # https://ieeexplore.ieee.org/document/9054421
    enc_hidden_size: int = 128
    dec_hidden_size: int = 128
