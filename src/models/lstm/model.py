import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2, pred_len=12):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.pred_len = pred_len

    def forward(self, batch, scene, scene_meta):
        _, _, obs_pos_rel, _, _, _ = batch
        B = obs_pos_rel.size(0)
        device = obs_pos_rel.device

        obs_pos_rel = obs_pos_rel.permute(0, 2, 1)

        # encoding
        _, (h, c) = self.encoder(obs_pos_rel)

        # decoding
        y = torch.zeros(B, 1, 2, device=device)
        pred_pos_rel = []
        for t in range(self.pred_len):
            out_t, (h, c) = self.decoder(y, (h, c))
            y = self.out(out_t)
            pred_pos_rel.append(y)

        return torch.cat(pred_pos_rel, dim=1).permute(0, 2, 1)

    def inference(self, batch, scene, scene_meta):
        return self.forward(batch, scene, scene_meta)
