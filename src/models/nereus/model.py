import torch
from torch import nn
from torch_geometric.data import Data

from data.map.rasterize import Rasterizer
from models.gru.modules.decoder import MDNDecoder
from models.gru.modules.encoder import GRUEncoder
from models.nereus.map_modules.map import ScenePoolingCNN
from models.nereus.params import NEREUSParams
from utils.config import TRAIN_BBOX


class NEREUS(nn.Module):
    """NEREUS model consisting of different Modules:
    - Encoder   (GRU)
    - Static    (linear)
    - Social    (GAT, SocialPooling)
    - Map       (ScenePoolingCNN, MapAttention)
    - Prior     (Traisformer, DensityMap, MAP_GMM)
    - Decoder   (MDN)
    """

    def __init__(
            self,
            config: NEREUSParams,
            static_module: bool = True,
            social_module = None,
            map_module = None,
            prior_module = None,
        ):
        super().__init__()
        assert not (map_module is not None and prior_module is None), \
            "map_module requires a prior_module to fuse its output"
        self.rasterizer = Rasterizer(TRAIN_BBOX, pos_res=config.map_res)

        # ENCODER
        self.encoder = GRUEncoder(config.rnn_hidden_size, config.node_feat_dim)
        self.enc_proj = nn.Linear(config.rnn_hidden_size, config.rnn_hidden_size)
        module_count = 1

        # STATIC
        self.static_module = static_module
        if self.static_module:
            self.static_proj = nn.Linear(config.static_feat_dim, config.rnn_hidden_size)
            module_count += 1

        # SOCIAL
        self.social_module = social_module
        if self.social_module:
            self.gnn_proj = nn.Linear(social_module.out_dim, config.rnn_hidden_size)
            module_count += 1

        # MAP
        self.map_cnn = map_module
        if self.map_cnn:
            self.map_proj = nn.Linear(config.map_cnn_out, config.rnn_hidden_size)
            module_count += 1

        # PRIOR
        self.prior_module = prior_module
        if self.prior_module:
            self.prior_cnn = ScenePoolingCNN(
                self.rasterizer,
                config=config,
                in_channels=1,
                out_channels=config.prior_cnn_out
            )
            self.prior_proj = nn.Linear(config.prior_cnn_out, config.rnn_hidden_size)
            module_count += 1

        # DECODER
        self.w = nn.Parameter(torch.tensor([1.0] * module_count))
        self.dropout_layer = nn.Dropout(0.1)
        self.decoder = MDNDecoder(config)

    def forward(self, data: Data, maps=None):
        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
        B = ego_idx.shape[0]
        abs_pos = data.x_pos[ego_idx, -1, :]
        rel_pos_t0 = data.x[ego_idx, -1:, :2]

        h_stack = []

        h_enc_all = self.encoder(data.x, data.x_mask)
        h_enc = self.enc_proj(h_enc_all[ego_idx])
        h_stack.append(h_enc.unsqueeze(0))

        if self.static_module:
            h_static = self.static_proj(data.static[ego_idx, :])
            h_stack.append(h_static.unsqueeze(0))

        if self.social_module:
            h_social = self.social_module(h_enc_all, data)
            h_social = self.gnn_proj(h_social[ego_idx])
            h_stack.append(h_social.unsqueeze(0))

        ### workaround to keep checkpoint working
        if self.prior_module:
            with torch.no_grad():
                prior_map, _ = self.prior_module(data, maps)
            h_prior = self.prior_cnn(prior_map, abs_pos)
            h_prior = self.prior_proj(h_prior)
        else:
            h_prior = None

        if self.map_cnn:
            maps_v = maps.unsqueeze(0).expand(B, -1, -1, -1)
            h_map = self.map_cnn(maps_v, abs_pos, h_prior)
            h_map = self.map_proj(h_map)
            h_stack.append(h_map.unsqueeze(0))

        if self.prior_module:
            h_stack.append(h_prior.unsqueeze(0))
        ###

        w = torch.softmax(self.w, dim=0)
        h_stack = torch.stack([torch.tanh(h) for h in h_stack], dim=0)
        h = torch.sum(w.view(-1, 1, 1, 1) * h_stack, dim=0)
        h = self.dropout_layer(h)

        return self.decoder(rel_pos_t0, h)

    def inference(self, data: Data, maps=None):
        return self.forward(data, maps)
