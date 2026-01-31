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
        self.encoder = LSTM_Encoder(config)

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

        h, _ = self.encoder(data)

        h = self.gnn(h, data.edge_index, data.edge_attr)
        h = F.relu(h)

        out = self.mlp(h)
        out = out.view(-1, self.pred_len, 2)

        return out
    

class LSTM(nn.Module):
    def __init__(self, config: GNNLSTMParams):
        super().__init__()

        # Temporal encoder (per node)
        self.encoder = LSTM_Encoder(config)

        # Decoder (predict future)
        self.mlp = nn.Sequential(
            nn.Linear(config.enc_hidden_size, config.dec_hidden_size),
            nn.ReLU(),
            nn.Linear(config.dec_hidden_size, config.pred_len * 2),
        )

        self.pred_len = config.pred_len

    def forward(self, data: Data, scene):
        """
        data.x: [N, obs_len, feat_dim]
        data.edge_index: [2, E]
        """

        h, _ = self.encoder(data)

        out = self.mlp(h)
        out = out.view(-1, self.pred_len, 2)

        return out
    


class Seq2SeqLSTM(nn.Module):
    def __init__(self, config: GNNLSTMParams):
        super().__init__()

        self.pred_len = config.pred_len

        # -------- Encoder --------
        self.encoder = LSTM_Encoder(config)

        self.h_proj = nn.Linear(config.enc_hidden_size, config.dec_hidden_size)
        self.c_proj = nn.Linear(config.enc_hidden_size, config.dec_hidden_size)

        # -------- Decoder --------
        self.decoder = nn.LSTM(
            input_size=2,
            hidden_size=config.dec_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(config.dec_hidden_size, 2)

    def forward(self, data: Data, scene=None):
        """
        data.x: [N, obs_len, feat_dim]
        """

        h, c = self.encoder(data)

        h = self.h_proj(h).unsqueeze(0)
        c = self.c_proj(c).unsqueeze(0)


        y = data.x[:, -1:, :2]  # assumes first two dims are position
        preds = []

        for _ in range(self.pred_len):
            out, (h, c) = self.decoder(y, (h, c))
            y = self.out(out)        # next position
            preds.append(y)

        preds = torch.cat(preds, dim=1)  # [N, pred_len, 2]

        return preds


class Seq2SeqGNNLSTM(nn.Module):
    def __init__(self, config: GNNLSTMParams):
        super().__init__()

        self.pred_len = config.pred_len

        # -------- Temporal encoder --------
        self.encoder = LSTM_Encoder(config)

        # -------- Interaction module (GNN) --------
        self.gnn = GATConv(
            in_channels=config.enc_hidden_size,
            out_channels=config.gnn_hidden_size,
            heads=config.gnn_n_head,
            edge_dim=config.edge_feat_dim,
            concat=False,
        )

        # -------- Bridge encoder → decoder --------
        self.h_proj = nn.Linear(config.gnn_hidden_size, config.dec_hidden_size)
        self.c_proj = nn.Linear(config.gnn_hidden_size, config.dec_hidden_size)

        # -------- Decoder --------
        self.decoder = nn.LSTM(
            input_size=2,  # (x, y) or (dx, dy)
            hidden_size=config.dec_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(config.dec_hidden_size, 2)

    def forward(self, data: Data, scene=None):
        """
        data.x: [N, obs_len, feat_dim]
        data.edge_index: [2, E]
        data.edge_attr: [E, edge_feat_dim] (optional)
        """

        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        # -------- Encode temporal history --------
        h_enc, _ = self.encoder(x)

        # -------- Apply GNN interaction --------
        h_gnn = self.gnn(h_enc, edge_index, edge_attr)

        # -------- Initialize decoder --------
        h = self.h_proj(h_gnn).unsqueeze(0)  # [1, N, dec_hidden]
        c = self.c_proj(h_gnn).unsqueeze(0)

        # -------- Autoregressive decoding --------
        y = x[:, -1:, :2]  # last observed position
        preds = []

        for _ in range(self.pred_len):
            out, (h, c) = self.decoder(y, (h, c))
            y = self.out(out)
            preds.append(y)

        preds = torch.cat(preds, dim=1)  # [N, pred_len, 2]

        return preds
    
class LSTM_Encoder(nn.Module):
    def __init__(self, config: GNNLSTMParams):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=config.node_feat_dim,
            hidden_size=config.enc_hidden_size,
            num_layers=config.enc_n_layers,
            batch_first=True,
        )

    def forward(self, data):
        x = data.x

        lengths = data.obs_mask.sum(dim=1).cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        _, (h_enc, c_enc) = self.encoder(packed_x)

        return h_enc[-1], c_enc[-1]