import torch
import torch.nn as nn

from models.dae.params import DAEParams
from models.dae.transformer_block import Block

class TransformerEncoder(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config: DAEParams):
        super().__init__()
        # embeddings
        self.ctx_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.traj_embd = nn.Linear(config.n_traj_feat, config.d_model)
        self.vessel_embd = nn.Linear(config.n_vessel_feat, config.d_model)
        self.terrain_embd = nn.Linear(64, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.obs_len + 2, config.d_model))
        self.drop = nn.Dropout(config.dropout)

        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.latent = nn.Linear(config.d_model, config.latent_dim)

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


    def forward(self, traj_feat, static_feat, mask):
        B, seqlen, _ = traj_feat.shape

        # embedding of context
        ctx = self.ctx_token.expand(B, -1, -1)
        traj_embeddings = self.traj_embd(traj_feat)
        vessel_embedding = self.vessel_embd(static_feat).unsqueeze(1)

        tokens = torch.cat([ctx, vessel_embedding, traj_embeddings], dim=1)

        position_embeddings = self.pos_emb[:, :seqlen + 2, :]
        fea = self.drop(tokens + position_embeddings)

        ctx_mask = torch.ones(B, 1, device=mask.device, dtype=torch.bool)
        full_mask = torch.cat([ctx_mask, ctx_mask, mask], dim=1)

        for blk in self.blocks:
            fea = blk(fea, full_mask)
        fea = self.ln_f(fea)

        z = fea[:, 0, :]
        latent = self.latent(z)
        return latent