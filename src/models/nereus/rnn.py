import torch
import torch.nn as nn
from models.nereus.params import NEREUSParams
from torch_geometric.data import Data

class MDNDecoder(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.pred_len = config.pred_len
        self.dec_hidden_size = config.dec_hidden_size

        self.decoder = nn.GRU(
            input_size=2,
            hidden_size=self.dec_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.num_modes = config.mdn_modes
        self.mdn = nn.Linear(self.dec_hidden_size, self.num_modes * 5)


    def forward(self, y, h):
        mdn_outputs = []
        for _ in range(self.pred_len):
            out, h = self.decoder(y, h)
            mdn_t = self.mdn(out)
            mdn_outputs.append(mdn_t)

            pi, mu = self.unpack_mdn(mdn_t)
            y = torch.sum(pi.unsqueeze(-1) * mu, dim=2)

        mdn_outputs = torch.cat(mdn_outputs, dim=1)
        return mdn_outputs

    def unpack_mdn(self, mdn_out):
        B, _, _ = mdn_out.shape
        K = self.num_modes

        mdn_out = mdn_out.view(B, 1, K, 5)

        pi = torch.softmax(mdn_out[..., 0], dim=-1)
        mu = mdn_out[..., 1:3]

        return pi, mu


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
        x = data.x

        lengths = data.x_mask.sum(dim=1).cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_enc = self.encoder(packed_x)

        return h_enc[-1]

class GRU(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.encoder = GRUEncoder(config)
        self.h_proj = nn.Linear(config.enc_hidden_size, config.dec_hidden_size)
        self.decoder = GRUDecoder(config)
        

    def forward(self, data: Data, scene=None):
        h = self.encoder(data)
        h = self.h_proj(h).unsqueeze(0)

        y = data.x[:, -1:, :2]  # [N, 1, 2]
        preds = self.decoder(y, h)

        return preds
    
class GRUDecoder(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.pred_len = config.pred_len
        self.decoder = nn.GRU(2, config.dec_hidden_size, batch_first=True)
        self.out = nn.Linear(config.dec_hidden_size, 2)

    def forward(self, y, h):
        preds = []
        for _ in range(self.pred_len):
            out, h = self.decoder(y, h)
            y = self.out(out)
            preds.append(y)

        return torch.cat(preds, dim=1)

class LSTM(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()

        self.encoder = LSTMEncoder(config)
        self.h_proj = nn.Linear(config.enc_hidden_size, config.dec_hidden_size)
        self.c_proj = nn.Linear(config.enc_hidden_size, config.dec_hidden_size)

        self.decoder = LSTMDecoder(config)
       

    def forward(self, data: Data, scene=None):
        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
    
        h, c = self.encoder(data)
        h = self.h_proj(h[ego_idx]).unsqueeze(0)
        c = self.c_proj(c[ego_idx]).unsqueeze(0)

        y = data.x[ego_idx, -1:, :2]
        preds = self.decoder(y, h, c)

        return preds
    
class LSTMDecoder(nn.Module):
    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.pred_len = config.pred_len
        self.decoder = nn.LSTM(
            input_size=2,
            hidden_size=config.dec_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(config.dec_hidden_size, 2)
    
    def forward(self, y, h, c):
        preds = []
        for _ in range(self.pred_len):
            out, (h, c) = self.decoder(y, (h, c))
            y = self.out(out)
            preds.append(y)

        return torch.cat(preds, dim=1)

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

        lengths = data.x_mask.sum(dim=1).cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_enc, c_enc) = self.encoder(packed_x)

        return h_enc[-1], c_enc[-1]