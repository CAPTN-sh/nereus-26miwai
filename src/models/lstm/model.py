import torch
import torch.nn as nn

from .params import LSTMParams

class LSTMModel(nn.Module):
    def __init__(self, config: LSTMParams):
        super().__init__()
        self.pred_len = config.pred_len
        self.encoder = nn.LSTM(2 + 11, config.enc_hidden_size, batch_first=True)

        self.h_proj = nn.Linear(config.enc_hidden_size, config.dec_hidden_size)
        self.c_proj = nn.Linear(config.enc_hidden_size, config.dec_hidden_size)

        self.decoder = nn.LSTM(2, config.dec_hidden_size, batch_first=True)
        self.out = nn.Linear(config.dec_hidden_size, 2)
        

    def forward(self, batch, scene=None):
        obs_feat, _, obs_rel, obs_mask, *_ = batch
        max_dist = 100 # 40kn for 5s

        x = torch.cat([obs_rel/max_dist, obs_feat], dim=-1)

        lengths = obs_mask.sum(dim=1).cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        _, (h, c) = self.encoder(packed_x)

        h = self.h_proj(h)
        c = self.c_proj(c)

        # decoding
        y = obs_rel[:, -1:, :]
        pred_rel = []
        for t in range(self.pred_len):
            output, (h, c) = self.decoder(y, (h, c))
            y = self.out(output)
            pred_rel.append(y)

        pred_rel = torch.cat(pred_rel, dim=1) * max_dist
        return pred_rel

    def inference(self, batch, scene=None):
        return self.forward(batch, scene), None
