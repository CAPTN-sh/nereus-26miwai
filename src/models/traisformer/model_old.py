import torch
import torch.nn as nn

from .params import TraisformerParams
from .rasterize import Rasterizer
from .transformer_block import Block


class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config: TraisformerParams):
        super().__init__()
        config.n_embd = 2*config.n_pos_embd + 2*config.n_kin_embd

        self.raster = Rasterizer(config.bbox)
        self.x_size, self.y_size, self.sog_size, self.cog_size = (
            self.raster.get_total_grid_sizes()
        )

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.x_size, config.n_pos_embd)
        self.lon_emb = nn.Embedding(self.y_size, config.n_pos_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_kin_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_kin_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.obs_len, config.n_embd))
        self.drop = nn.Dropout(config.dropout)

        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # pooling
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.intent_head = nn.Linear(config.n_embd, self.x_size * self.y_size)

        self.apply(self._init_weights)

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
        sog_idx, cog_idx = self.raster.kin_to_index(obs_feat[..., :2])

        token_embeddings = torch.cat(
            (
                self.lat_emb(x_idx),
                self.lon_emb(y_idx),
                self.sog_emb(sog_idx),
                self.cog_emb(cog_idx),
            ), dim=-1
        )

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
