import torch
import torch.nn as nn

from .params import LSTMParams


class LSTMModel(nn.Module):
    def __init__(self, config: LSTMParams):
        super().__init__()
        self.encoder = nn.LSTM(config.input_size, config.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(2, config.hidden_size, batch_first=True)
        self.out = nn.Linear(config.hidden_size, 2)
        self.pred_len = config.pred_len

    def forward(self, batch, scene, scene_meta):
        obs_feat, _, obs_pos_rel, _, _, _, _ = batch
        B = obs_pos_rel.size(0)
        device = obs_pos_rel.device

        x = torch.cat([obs_feat, obs_pos_rel], dim=1)
        x = x.permute(0, 2, 1)

        # encoding
        _, (h, c) = self.encoder(x)

        # decoding
        y = torch.zeros(B, 1, 2, device=device)
        pred_pos_rel = []
        for t in range(self.pred_len):
            out_t, (h, c) = self.decoder(y, (h, c))
            y = self.out(out_t)
            pred_pos_rel.append(y)

        return torch.cat(pred_pos_rel, dim=1).permute(0, 2, 1)

    def inference(self, batch, scene, scene_meta):
        return self.forward(batch, scene, scene_meta), None
