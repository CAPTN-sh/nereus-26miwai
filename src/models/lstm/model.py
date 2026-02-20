import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nereus.params import NEREUSParams
from models.utils.rnn import GRUEncoder, GRUDecoder
from torch_geometric.data import Data


class RNN(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.encoder = GRUEncoder(config.rnn_hidden_size, config.node_feat_dim)
        self.h_proj = nn.Linear(config.rnn_hidden_size, config.rnn_hidden_size)
        self.decoder = GRUDecoder(config)
        

    def forward(self, data: Data, scene=None):
        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]

        h = self.encoder(data.x[ego_idx], data.x_mask[ego_idx])
        h = self.h_proj(h).unsqueeze(0)

        y = data.x[ego_idx, -1:, :2]
        preds = self.decoder(y, h)

        return preds