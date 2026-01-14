import torch
import torch.nn as nn
import torch.nn.functional as F

from models.desire.utils.params import DESIREParams


class RNNEncoder(nn.Module):
    def __init__(self, params: DESIREParams, kernel_size, input_dim):
        super(RNNEncoder, self).__init__()
        _c = params.intermediate_size
        _h = params.hidden_size
        _in = input_dim

        self.in_conv1d = nn.Conv1d(_in, _c, kernel_size = kernel_size, padding = kernel_size // 2)
        self.gru = nn.GRU(_c, _h, batch_first=True)

    def forward(self, x_seq, mask, hidden=None):
        x = F.relu(self.in_conv1d(x_seq))
        x = x.transpose(1, 2).contiguous()

        lengths = mask.sum(dim=-1).squeeze(-1).long()
        valid = lengths > 0
        lengths = torch.clamp(lengths, min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed, hidden)
        return hidden * valid.unsqueeze(0).unsqueeze(-1)


class RNNDecoder(nn.Module):
    def __init__(self, params: DESIREParams):
        super().__init__()
        _h = params.hidden_size
        _out = params.pred_dim

        self.gru = nn.GRU(_h, _h, batch_first=True)
        self.dec_fc = nn.Linear(_h, _out)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        pos = self.dec_fc(out)
        return pos, hidden
