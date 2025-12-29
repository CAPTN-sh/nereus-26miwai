import torch
import torch.nn as nn

from .params import TraisformerParams
from .rasterize import Rasterizer
from .transformer_block import Block
from .map_encoder import ScenePoolingCNN


class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config: TraisformerParams):
        super().__init__()
        self.config = config
        self.raster = Rasterizer(config.bbox)
        self.x_size, self.y_size, sog_size, cog_size, acc_size, rot_size = (
            self.raster.get_total_grid_sizes()
        )

        # state_embd
        self.lat_emb = nn.Embedding(self.x_size, config.n_spatial_embd)
        self.lon_emb = nn.Embedding(self.y_size, config.n_spatial_embd)
        if config.n_kinematic_embd > 0:
            self.sog_emb = nn.Embedding(sog_size, config.n_kinematic_embd)
            self.cog_emb = nn.Embedding(cog_size, config.n_kinematic_embd)
        if config.n_dynamic_embd > 0:
            self.acc_emb = nn.Embedding(acc_size, config.n_dynamic_embd)
            self.rot_emb = nn.Embedding(rot_size, config.n_dynamic_embd)
        # context_embd
        if config.n_terrain_embd > 0:
            self.scene_cnn = ScenePoolingCNN(in_channels=4)
            self.terrain_embd = nn.Linear(64, config.n_terrain_embd)
        if config.n_vessel_embd > 0:
            self.vessel_embd = nn.Linear(config.n_vessel_feat, config.n_vessel_embd)

        # config.n_embd = 2*config.n_spatial_embd + 2*config.n_kinematic_embd + 2*config.n_dynamic_embd + config.n_terrain_embd + config.n_vessel_embd

        self.pos_emb = nn.Parameter(torch.zeros(1, config.obs_len, config.n_embd))
        self.drop = nn.Dropout(config.dropout)

        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # pooling
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.intent_head = nn.Linear(config.n_embd, self.x_size * self.y_size)

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch, scene=None, scene_meta=None):
        obs_feat, obs_pos, _, obs_mask, *_ = batch
        B, seqlen, _ = obs_pos.size()

        x_idx, y_idx = self.raster.pos_to_index(obs_pos)
        embeddings_list = [
            self.lat_emb(x_idx),
            self.lon_emb(y_idx)
        ]

        if self.config.n_kinematic_embd > 0:
            sog_idx, cog_idx = self.raster.kin_to_index(obs_feat[..., :2])
            embeddings_list.extend([self.sog_emb(sog_idx), self.cog_emb(cog_idx)])

        if self.config.n_dynamic_embd > 0:
            acc_idx, rot_idx = self.raster.dyn_to_index(obs_feat[..., 2:4])
            embeddings_list.extend([self.acc_emb(acc_idx), self.rot_emb(rot_idx)])

        if self.config.n_terrain_embd > 0:
            map_features = self.scene_cnn(scene).unsqueeze(1).expand(B, seqlen, -1)
            embeddings_list.append(self.terrain_embd(map_features))

        if self.config.n_vessel_embd > 0:
            embeddings_list.append(self.vessel_embd(obs_feat[..., 4:]))

        token_embeddings = torch.cat(embeddings_list, dim=-1)

        position_embeddings = self.pos_emb[:, :seqlen, :]
        fea = self.drop(token_embeddings + position_embeddings)
        for blk in self.blocks:
            fea = blk(fea, obs_mask)
        fea = self.ln_f(fea)

        z = fea[:, -1, :]
        logits_flat = self.intent_head(z)
        logits = logits_flat.view(B, 1, self.x_size, self.y_size)

        return {"intent_logits": logits}

    def inference(self, batch, scene=None, scene_meta=None):
        return self.forward(batch, scene, scene_meta), None
