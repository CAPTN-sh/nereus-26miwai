import torch
import torch.nn as nn

from .params import TraisformerParams
from .rasterize import Rasterizer
from .transformer_block import Block
from .upsampling import UpsamplingDecoder


class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config: TraisformerParams):
        super().__init__()

        self.raster = Rasterizer(config)
        self.x_size, self.y_size, self.sog_size, self.cog_size = (
            self.raster.get_total_gird_sizes()
        )

        self.n_x_embd = 128
        self.n_y_embd = 128
        self.n_sog_embd = 64
        self.n_cog_embd = 64
        self.n_emdb = self.n_x_embd + self.n_y_embd + self.n_sog_embd + self.n_cog_embd

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.x_size, self.n_x_embd)
        self.lon_emb = nn.Embedding(self.y_size, self.n_y_embd)
        self.sog_emb = nn.Embedding(self.sog_size, self.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, self.n_cog_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, self.n_emdb))
        self.drop = nn.Dropout(config.pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # pooling
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.query = nn.Parameter(torch.randn(1, 1, config.n_embd))
        self.mha = nn.MultiheadAttention(config.n_embd, 4, batch_first=True)

        # decoder head and up conv heat map
        self.upsampeling = UpsamplingDecoder(config.n_embd, self.x_size, self.y_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, scene=None, scene_meta=None):
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
        Returns:
            logits, loss
        """

        device = next(self.parameters()).device
        batch = [t.to(device) if torch.is_tensor(t) else t for t in x]
        # obs_feat, obs_pos, obs_pos_rel, fut_pos, fut_pos_rel
        obs_feat, obs_pos, _, _, _, _ = batch

        x_idx, y_idx = self.raster.pos_to_index(obs_pos)
        sog_idx, cog_idx = self.raster.feat_to_index(obs_feat)

        # forward the GPT model
        lat_embeddings = self.lat_emb(x_idx)  # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(y_idx)
        sog_embeddings = self.sog_emb(sog_idx)
        cog_embeddings = self.cog_emb(cog_idx)

        token_embeddings = torch.cat(
            (lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings), dim=-1
        )

        B, _, seqlen = obs_pos.size()
        position_embeddings = self.pos_emb[
            :, :seqlen, :
        ]  # each position maps to a (learnable) vector (1, seqlen, n_embd)
        fea = self.drop(token_embeddings + position_embeddings)
        fea = self.blocks(fea)
        fea = self.ln_f(fea)

        q = self.query.expand(B, 1, self.n_emdb)
        z, _ = self.mha(q, fea, fea)

        logits = self.upsampeling(z)

        return {"intent_logits": logits}

    def inference(self, batch, scene=None, scene_meta=None):
        return self.forward(batch, scene, scene_meta), None
