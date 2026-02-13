import torch
import torch.nn as nn

from models.traisformer.params import TraisformerParams
from models.utils.maps.rasterize import Rasterizer
from models.traisformer.transformer_block import Block
from models.traisformer.map_encoder import ScenePoolingCNN
from models.traisformer import heatmap_head

HEAD_REGISTRY = {
    "linear": heatmap_head.LinearHead,
    "factorized": heatmap_head.FactorizedHead,
    "cnn": heatmap_head.CNNHead,
    "mixture": heatmap_head.MixtureHead,
    "lowrank": heatmap_head.LowRankHead,
}

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
        self.lat_emb = nn.Embedding(self.x_size, config.n_embd // 4)
        self.lon_emb = nn.Embedding(self.y_size, config.n_embd // 4)
        self.sog_emb = nn.Embedding(sog_size, config.n_embd // 8)
        self.cog_emb = nn.Embedding(cog_size, config.n_embd // 8)
        self.acc_emb = nn.Embedding(acc_size, config.n_embd // 8)
        self.rot_emb = nn.Embedding(rot_size, config.n_embd // 8)

        #self.scene_cnn = ScenePoolingCNN(in_channels=4)
        #self.terrain_embd = nn.Linear(64, config.n_embd)
        #self.vessel_embd = nn.Linear(config.n_vessel_feat, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.obs_len, config.n_embd)) #  + 2
        self.drop = nn.Dropout(config.dropout)

        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # pooling
        self.ln_f = nn.LayerNorm(config.n_embd)
        # LinearHead, FactorizedHead, CNNHead, MixtureHead
        head_cls = HEAD_REGISTRY[config.intent_head]
        self.intent_head = head_cls(config.n_embd, self.x_size, self.y_size, config.k_rank)

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

    def forward(self, data, scene=None):
        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
        obs_feat = data.x_raw[ego_idx, :, :]
        obs_pos = data.x_pos[ego_idx, :, :]
        static = data.static[ego_idx, :]
        obs_mask = data.x_mask[ego_idx, :]
        B, seqlen, _ = obs_pos.shape

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
        #map_embedding = self.terrain_embd(self.scene_cnn(scene.unsqueeze(0))).expand(B, -1).unsqueeze(1)
        #vessel_embedding = self.vessel_embd(static).unsqueeze(1)

        #tokens = torch.cat([map_embedding, vessel_embedding, traj_embeddings], dim=1)

        position_embeddings = self.pos_emb[:, :seqlen, :] #  + 2
        fea = self.drop(traj_embeddings + position_embeddings) # tokens

        #ctx_mask = torch.ones(B, 1, device=obs_mask.device, dtype=torch.bool)
        #full_mask = torch.cat([ctx_mask, ctx_mask, obs_mask], dim=1)

        for blk in self.blocks:
            fea = blk(fea, obs_mask) # full_mask
        fea = self.ln_f(fea)

        z = fea[:, -1, :] # 0
        logits = self.intent_head(z)

        return logits

    def inference(self, batch, scene=None):
        return self.forward(batch, scene), None
