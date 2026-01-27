import torch
import torch.nn as nn

from models.traisformer.params import TraisformerParams
from models.utils.maps.rasterize import Rasterizer
from models.traisformer.transformer_block import Block
from models.traisformer.map_encoder import ScenePoolingCNN


class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config: TraisformerParams):
        super().__init__()
        self.config = config
        self.rasterizer = Rasterizer(config.bbox)
        self.x_size, self.y_size, sog_size, cog_size, acc_size, rot_size = (
            self.rasterizer.get_total_grid_sizes()
        )

        # state_embd
        self.lat_emb = nn.Embedding(self.x_size, config.n_spatial_embd)
        self.lon_emb = nn.Embedding(self.y_size, config.n_spatial_embd)
        self.sog_emb = nn.Embedding(sog_size, config.n_kinematic_embd)
        self.cog_emb = nn.Embedding(cog_size, config.n_kinematic_embd)
        self.acc_emb = nn.Embedding(acc_size, config.n_dynamic_embd)
        self.rot_emb = nn.Embedding(rot_size, config.n_dynamic_embd)

        #config.n_embd = 2*config.n_spatial_embd + 2*config.n_kinematic_embd + 2*config.n_dynamic_embd

        self.scene_cnn = ScenePoolingCNN(in_channels=4)
        self.terrain_embd = nn.Linear(64, config.n_terrain_embd)
        self.vessel_embd = nn.Linear(config.n_vessel_feat, config.n_vessel_embd)
        self.ctx_proj = nn.Linear(config.n_terrain_embd + config.n_vessel_embd, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.obs_len + 1, config.n_embd))
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

    def forward(self, batch, scene=None):
        obs_feat, obs_pos, _, obs_mask, *_ = batch
        B, seqlen, _ = obs_pos.size()

        # embedding of traj
        x_idx, y_idx = self.rasterizer.pos_to_index(obs_pos)
        sog_idx, cog_idx = self.rasterizer.kin_to_index(obs_feat[..., :2])
        acc_idx, rot_idx = self.rasterizer.dyn_to_index(obs_feat[..., 2:4])

        traj_embeddings = torch.cat([
            self.lat_emb(x_idx), self.lon_emb(y_idx),
            self.sog_emb(sog_idx), self.cog_emb(cog_idx),
            self.acc_emb(acc_idx), self.rot_emb(rot_idx),
        ], dim=-1)

        # embedding of context
        map_embedding = self.terrain_embd(self.scene_cnn(scene)).expand(B, -1)
        vessel_embedding = self.vessel_embd(obs_feat[:, -1, 4:])
        context_embedding = torch.cat([map_embedding, vessel_embedding], dim=-1)
        context_token = self.ctx_proj(context_embedding).unsqueeze(1)

        tokens = torch.cat([context_token, traj_embeddings], dim=1)

        position_embeddings = self.pos_emb[:, :seqlen + 1, :]
        fea = self.drop(tokens + position_embeddings)

        ctx_mask = torch.ones(B, 1, device=obs_mask.device, dtype=torch.bool)
        full_mask = torch.cat([ctx_mask, obs_mask], dim=1)

        for blk in self.blocks:
            fea = blk(fea, full_mask)
        fea = self.ln_f(fea)

        z = fea[:, 0, :]
        logits_flat = self.intent_head(z)
        logits = logits_flat.view(B, 1, self.x_size, self.y_size)

        return {"intent_logits": logits}

    def inference(self, batch, scene=None):
        return self.forward(batch, scene), None
