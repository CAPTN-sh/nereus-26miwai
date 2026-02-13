import torch
import torch.nn as nn
import torch.nn.functional as F

from models.desire.utils.params import DESIREParams
    
class GRUDecoder(nn.Module):
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