import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from models.gnn.params import GNNLSTMParams

from torch_geometric.data import Data

class GNNLSTM(nn.Module):
    def __init__(self, config: GNNLSTMParams):
        super().__init__()

        # Temporal encoder (per node)
        self.lstm = nn.LSTM(
            input_size=config.node_feat_dim,
            hidden_size=config.enc_hidden_size,
            num_layers=config.enc_n_layers,
            batch_first=True,
        )

        # 2Interaction module (graph)
        self.gnn = GATConv(
            config.enc_hidden_size,
            config.gnn_hidden_size,
            heads=config.gnn_n_head,
            edge_dim=config.edge_feat_dim,
            concat=False,
        )

        # Decoder (predict future)
        self.mlp = nn.Sequential(
            nn.Linear(config.gnn_hidden_size, config.dec_hidden_size),
            nn.ReLU(),
            nn.Linear(config.dec_hidden_size, config.pred_len * 2),
        )

        self.pred_len = config.pred_len

    def forward(self, data: Data, scene):
        """
        data.x: [N, obs_len, feat_dim]
        data.edge_index: [2, E]
        """

        x = data.x

        _, (h, _) = self.lstm(x)
        h = h[-1]

        h = self.gnn(h, data.edge_index, data.edge_attr)
        h = F.relu(h)

        out = self.mlp(h)
        out = out.view(-1, self.pred_len, 2)

        return out