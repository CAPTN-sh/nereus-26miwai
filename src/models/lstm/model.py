import torch
import torch.nn as nn

from .params import LSTMParams
from models.traisformer.rasterize import Rasterizer

class LSTMModel(nn.Module):
    def __init__(self, config: LSTMParams):
        super().__init__()
        self.pred_len = config.pred_len
        raster = Rasterizer([10.12, 54.31, 10.33, 54.46])
        self.x_min = raster.x_min
        self.y_min = raster.y_min
        self.x_range = raster.x_max - raster.x_min
        self.y_range = raster.y_max - raster.y_min

        self.encoder = nn.LSTM(4, config.enc_hidden_size, batch_first=True)

        self.h_proj = nn.Linear(config.enc_hidden_size, config.dec_hidden_size)
        self.c_proj = nn.Linear(config.enc_hidden_size, config.dec_hidden_size)

        self.decoder = nn.LSTM(2, config.dec_hidden_size, batch_first=True)
        self.out = nn.Linear(config.dec_hidden_size, 2)
        

    def forward(self, batch, scene, scene_meta):
        obs_feat, _, obs_rel, *_ = batch
        max_dist = 100 # 40kn for 5s
        x = torch.cat([obs_feat, obs_rel/max_dist], dim=-1)

        _, (h, c) = self.encoder(x)

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

    def inference(self, batch, scene, scene_meta):
        return self.forward(batch, scene, scene_meta), None
