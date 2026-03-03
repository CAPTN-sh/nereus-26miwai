import torch
import torch.nn as nn
from models.nereus.params import NEREUSParams

class GRUDecoder(nn.Module):
    """
    Single layer GRU decoder for determenistic predictions.
    """
    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.pred_len = config.pred_len
        self.decoder = nn.GRU(2, config.rnn_hidden_size, batch_first=True)
        self.out = nn.Linear(config.rnn_hidden_size, 2)

    def forward(self, y, h):
        preds = []
        for _ in range(self.pred_len):
            out, h = self.decoder(y, h)
            y = self.out(out)
            preds.append(y)

        return torch.cat(preds, dim=1)

class MDNDecoder(nn.Module):
    """
    Single layer GRU decoder with a MDN-head for probabilistic predictions.
    """
    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.pred_len = config.pred_len
        self.hidden_size = config.rnn_hidden_size

        self.decoder = nn.GRU(
            input_size=2,
            hidden_size=self.hidden_size,
            batch_first=True,
        )

        self.num_modes = config.mdn_modes
        self.mdn = nn.Linear(self.hidden_size, self.num_modes * 5)

    def forward(self, y, h):
        mdn_outputs = []
        for _ in range(self.pred_len):
            out, h = self.decoder(y, h)
            mdn_t = self.mdn(out)
            mdn_outputs.append(mdn_t)

            pi, mu = self.unpack_mdn(mdn_t)
            y = torch.sum(pi.unsqueeze(-1) * mu, dim=2)

        mdn_outputs = torch.cat(mdn_outputs, dim=1)
        return mdn_outputs

    def unpack_mdn(self, mdn_out):
        B, _, _ = mdn_out.shape
        K = self.num_modes

        mdn_out = mdn_out.view(B, 1, K, 5)

        pi = torch.softmax(mdn_out[..., 0], dim=-1)
        mu = mdn_out[..., 1:3]

        return pi, mu