import torch
import torch.nn as nn

from models.dae.params import DAEParams
from models.dae.transformer_block import Block

class TransformerDecoder(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config: DAEParams):
        super().__init__()
        # embeddings
        self.ctx_proj = nn.Linear(config.latent_dim, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.obs_len, config.d_model))
        self.drop = nn.Dropout(config.dropout)

        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)

        self.output_head = nn.Linear(config.d_model, config.n_traj_feat)

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

    def forward(self, latent, mask):
        B, seqlen = mask.shape

        ctx_token = self.ctx_proj(latent)
        tokens = ctx_token.unsqueeze(1).repeat(1, seqlen, 1)

        position_embeddings = self.pos_emb[:, :seqlen, :]
        fea = self.drop(tokens + position_embeddings)

        for blk in self.blocks:
            fea = blk(fea, mask)
        fea = self.ln_f(fea)

        rec = self.output_head(fea)
        return rec