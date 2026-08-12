from torch import nn
from torch_geometric.nn import GATConv

from models.nereus.params import NEREUSParams
from models.nereus.social_modules.utils import SocialPoolFast


class GAT(nn.Module):
    """Graph attention network encoding social interations between the vessels.
    """

    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.out_dim = config.gnn_hidden_size
        self.pre_gnn_norm = nn.LayerNorm(config.rnn_hidden_size)
        self.gnn = GATConv(
            in_channels=config.rnn_hidden_size,
            out_channels=config.gnn_hidden_size,
            heads=config.gnn_n_head,
            edge_dim=config.edge_feat_dim,
            concat=False,
        )
        self.dropout_layer = nn.Dropout(0.1)

    def forward(self, h_enc, data):
        h_enc_norm = self.pre_gnn_norm(h_enc)
        h_gnn = self.gnn(h_enc_norm, data.edge_index, data.edge_attr)
        h_gnn = self.dropout_layer(h_gnn)

        return h_gnn

class SocialPooling(nn.Module):
    """Social pooling adapted from DESIRE to encode social interations through binning.
    """

    def __init__(self, config: NEREUSParams):
        super().__init__()
        self.out_dim = config.rnn_hidden_size
        self.social_pool = SocialPoolFast(config)
        self.pre_norm = nn.LayerNorm(config.rnn_hidden_size)

    def forward(self, h_enc, data):
        h_enc = self.pre_norm(h_enc)

        pooled = self.social_pool(
            y_pred=data.x_pos[:, -1:, :],        # [N, 1, 2]
            hidden=h_enc.unsqueeze(1),           # [N, 1, H]
            ego_idx=data.is_ego.nonzero(as_tuple=True)[0],
            edge_index=data.edge_index
        )                                        # [N, 1, H]

        return pooled.squeeze(1)
