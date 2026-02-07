import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, GATConv, GATv2Conv
from models.nereus.params import NEREUSParams

from torch_geometric.data import Data

class GRUEncoder(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=config.node_feat_dim,
            hidden_size=config.enc_hidden_size,
            num_layers=config.enc_n_layers,
            batch_first=True,
        )
        
    def forward(self, data: Data):
        x = data.x  # [N, obs_len, feat_dim]

        lengths = data.obs_mask.sum(dim=1).cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_enc = self.encoder(packed_x)

        # take last layer hidden state
        return h_enc[-1]

class GRU(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()

        self.pred_len = config.pred_len

        # -------- Encoder --------
        self.encoder = GRUEncoder(config)
        self.h_proj = nn.Linear(
            config.enc_hidden_size, config.dec_hidden_size
        )

        # -------- Decoder --------
        self.decoder = nn.GRU(
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

        h = self.encoder(data)
        h = self.h_proj(h).unsqueeze(0)  # [1, N, dec_hidden_size]

        # last observed position as initial input
        y = data.x[:, -1:, :2]  # [N, 1, 2]
        preds = []

        for _ in range(self.pred_len):
            out, h = self.decoder(y, h)
            y = self.out(out)
            preds.append(y)

        preds = torch.cat(preds, dim=1)  # [N, pred_len, 2]

        return preds


class LSTM(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()

        self.pred_len = config.pred_len

        # -------- Encoder --------
        self.encoder = LSTMEncoder(config)
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


class GNNLSTM(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()

        self.pred_len = config.pred_len

        # -------- Temporal encoder --------
        self.encoder = LSTMEncoder(config)
        self.dropout_layer = nn.Dropout(0.1)

        # -------- Interaction module (GNN) --------
        self.pre_gnn_norm = nn.LayerNorm(config.enc_hidden_size)
        self.gnn = GATConv(
            in_channels=config.enc_hidden_size,
            out_channels=config.gnn_hidden_size,
            heads=config.gnn_n_head,
            edge_dim=config.edge_feat_dim,
            concat=False,
        )
        self.gnn_proj = nn.Linear(config.gnn_hidden_size, config.enc_hidden_size)

        # -------- Bridge encoder → decoder --------
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
        data.edge_index: [2, E]
        data.edge_attr: [E, edge_feat_dim] (optional)
        """
        
        h_enc, c_enc = self.encoder(data)
        h_enc = self.dropout_layer(h_enc)
        h_enc_norm = self.pre_gnn_norm(h_enc)

        h_gnn = self.gnn(h_enc_norm, data.edge_index, data.edge_attr)
        h_gnn = self.dropout_layer(h_gnn)
        h_gnn = self.gnn_proj(h_gnn)

        h = torch.tanh(self.h_proj(h_gnn)).unsqueeze(0)
        c = torch.tanh(self.c_proj(c_enc)).unsqueeze(0)

        y = data.x[:, -1:, :2]
        preds = []

        for t in range(self.pred_len):
            out, (h, c) = self.decoder(y, (h, c))
            y = self.out(out)
            preds.append(y)

        preds = torch.cat(preds, dim=1)

        return preds
    

class LSTMEncoder(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=config.node_feat_dim,
            hidden_size=config.enc_hidden_size,
            num_layers=config.enc_n_layers,
            batch_first=True,
        )
        
    def forward(self, data: Data):
        x = data.x

        lengths = data.obs_mask.sum(dim=1).cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_enc, c_enc) = self.encoder(packed_x)

        return h_enc[-1], c_enc[-1]


class GNNTransformer(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()

        self.pred_len = config.pred_len

        # -------- Temporal encoder --------
        self.encoder = TransformerEncoder(config)
        self.dropout_layer = nn.Dropout(0.1)

        # -------- Interaction module (GNN) --------
        self.pre_gnn_norm = nn.LayerNorm(config.enc_hidden_size)
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
            input_size=2,
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
        
        h_enc = self.encoder(data)

        h_enc = self.dropout_layer(h_enc)
        h_enc_norm = self.pre_gnn_norm(h_enc)
        h_gnn = self.gnn(h_enc_norm, data.edge_index, data.edge_attr)

        h_gnn = self.dropout_layer(h_gnn)

        h = torch.tanh(self.h_proj(h_gnn)).unsqueeze(0)
        c = torch.tanh(self.c_proj(h_gnn)).unsqueeze(0)

        y = data.x[:, -1:, :2]
        preds = []

        for t in range(self.pred_len):
            out, (h, c) = self.decoder(y, (h, c))
            y = self.out(out)
            preds.append(y)

        preds = torch.cat(preds, dim=1)

        return preds 

class TransformerEncoder(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()
        # Project node features → model dimension
        self.input_proj = nn.Linear(
            config.node_feat_dim, config.enc_hidden_size
        )
        self.in_norm = nn.LayerNorm(config.enc_hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.enc_hidden_size,
            nhead=8,
            dim_feedforward=config.enc_hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.enc_n_layers)

        # Separate projection to mimic LSTM cell state
        self.c_proj = nn.Linear(config.enc_hidden_size, config.enc_hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.obs_len, config.enc_hidden_size))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, data: Data):
        x = data.x
        _, T, _ = x.shape

        x = self.in_norm(self.input_proj(x) + self.pos_emb[:, :T])
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h_seq = self.encoder(x, mask=attn_mask, src_key_padding_mask=~data.obs_mask.bool())

        mask = data.obs_mask.unsqueeze(-1)
        h_enc = (h_seq * mask).sum(dim=1) / mask.sum(dim=1)

        return h_enc