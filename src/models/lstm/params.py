from dataclasses import dataclass


@dataclass
class LSTMParams:
    # Sequence lengths
    # TODO from data loader config
    pred_len: int = 36
    input_size: int = 4

    # Model dims
    hidden_size: int = 64
