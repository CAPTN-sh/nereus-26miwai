import torch
from torch import nn

from data.map.rasterize import Rasterizer
from models.traisformer.modules.causal_block import CausalBlock, causal_padding_mask
from models.traisformer.params_ar import TraisformerARParams
from models.traisformer.tokenize import build_token_sequence, cells_to_positions
from utils.config import TRAIN_BBOX

POS_SCALE = 100.0


class TrAISformerAR(nn.Module):
    """TrAISformer as published (Nguyen & Fablet, arXiv:2109.03958).

    A causal transformer over the discretised AIS state: each timestep is four tokens
    (x cell, y cell, SOG bin, COG bin), embedded separately and concatenated, and the
    head emits one logit vector per attribute for the *next* timestep. Training is
    teacher-forced next-token cross-entropy; prediction is an autoregressive rollout,
    which is what makes ADE/FDE well defined for this variant -- unlike the heatmap
    model in ``model_heatmap.py``, which emits a single time-less occupancy map.

    Positions are absolute cells of the training bounding box, exactly as in the paper
    (that is how route structure gets learned), so a trained model is tied to its
    region: evaluating on another bbox needs a retrain, not just a rasterizer swap.
    """

    def __init__(self, config: TraisformerARParams):
        super().__init__()
        self.cfg = config
        self.rasterizer = Rasterizer(TRAIN_BBOX, pos_res=config.pos_res)
        x_size, y_size, sog_size, cog_size, *_ = self.rasterizer.get_total_grid_sizes()
        self.att_sizes = (x_size, y_size, sog_size, cog_size)

        n_embd = config.n_x_embd + config.n_y_embd + config.n_sog_embd + config.n_cog_embd
        assert n_embd == config.n_embd, (
            f"n_embd ({config.n_embd}) must equal the sum of the attribute embeddings ({n_embd})"
        )

        self.x_emb = nn.Embedding(x_size, config.n_x_embd)
        self.y_emb = nn.Embedding(y_size, config.n_y_embd)
        self.sog_emb = nn.Embedding(sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(cog_size, config.n_cog_embd)

        seq_len = config.obs_len + config.pred_len
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([CausalBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, sum(self.att_sizes), bias=False)

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

    def _check_grid(self):
        if self.rasterizer.x_size != self.x_emb.num_embeddings or \
           self.rasterizer.y_size != self.y_emb.num_embeddings:
            raise ValueError(
                f"Rasterizer grid {self.rasterizer.x_size}x{self.rasterizer.y_size} does not "
                f"match the embeddings {self.x_emb.num_embeddings}x{self.y_emb.num_embeddings}. "
                "TrAISformerAR uses absolute position cells, so it cannot be evaluated on a "
                "region other than the one it was trained on."
            )

    def logits(self, idx, mask, cache=None, pos_offset: int = 0):
        """``idx`` [B, L, 4] -> per-attribute logits [B, L, sum(att_sizes)].

        ``mask`` marks valid *keys*: the current window when ``cache`` is None, the
        whole cached sequence when generating one token at a time.
        """
        self._check_grid()
        tokens = torch.cat([
            self.x_emb(idx[..., 0]), self.y_emb(idx[..., 1]),
            self.sog_emb(idx[..., 2]), self.cog_emb(idx[..., 3]),
        ], dim=-1)
        L = idx.size(1)
        x = self.drop(tokens + self.pos_emb[:, pos_offset:pos_offset + L, :])
        # A single query attends to every cached key, so validity alone is enough;
        # a multi-token window still needs the causal mask, cached or not.
        attn_mask = mask[:, None, None, :] if L == 1 else causal_padding_mask(mask)
        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask, None if cache is None else cache[i])
        return self.head(self.ln_f(x))

    def split_logits(self, logits):
        return torch.split(logits, self.att_sizes, dim=-1)

    def forward(self, data, scene=None):
        """Teacher-forced pass. Returns ``(logits, targets, mask, att_sizes)``.

        The sequence is the observation window followed by the ego future, and every
        position predicts the next one, so the model is trained on the same next-token
        objective inside and beyond the observation window.
        """
        idx, mask, _ = build_token_sequence(data, self.rasterizer, include_future=True)
        inputs, targets = idx[:, :-1], idx[:, 1:]
        target_mask = mask[:, 1:] & mask[:, :-1]
        return self.logits(inputs, mask[:, :-1]), targets, target_mask, self.att_sizes

    def _restrict_to_vicinity(self, logits, current, size):
        """Zero out cells more than ``r_vicinity`` away from the current one.

        The paper's ``pos_vicinity`` sampling mode: a vessel cannot jump across the map
        in one step, and without this an under-trained model happily samples from the
        tail and derails the whole rollout.
        """
        if self.cfg.sample_mode != "vicinity":
            return logits
        cells = torch.arange(size, device=logits.device)
        allowed = (cells.unsqueeze(0) - current.unsqueeze(1)).abs() <= self.cfg.r_vicinity
        return logits.masked_fill(~allowed, float("-inf"))

    def _next_token(self, logits, greedy_rows):
        """Argmax for the greedy rows, temperature/top-k sampling for the rest."""
        probs_src = logits / self.cfg.temperature
        if self.cfg.top_k is not None:
            k = min(self.cfg.top_k, probs_src.size(-1))
            kth = torch.topk(probs_src, k, dim=-1).values[..., -1:]
            probs_src = probs_src.masked_fill(probs_src < kth, float("-inf"))
        sampled = torch.multinomial(torch.softmax(probs_src, dim=-1), 1).squeeze(-1)
        return torch.where(greedy_rows, logits.argmax(dim=-1), sampled)

    @torch.no_grad()
    def inference(self, data, scene=None):
        """Autoregressive rollout. Returns ``(best_rel [B, T, 2], k_rel [B, K, T, 2])``.

        Row 0 of the rollout becomes ``best_rel``, the trajectory behind the ADE/FDE
        columns; the remaining ``num_samples`` rows are always sampled and feed the
        min-over-K columns. By default row 0 is greedy (argmax) -- matching the
        paper's "TrAISformer_No-Stoch" ablation -- but ``cfg.greedy_best=False`` makes
        it a single stochastic sample instead. Either way this is one draw, not
        best-of-N: the paper's headline best-of-N numbers correspond to
        ``k_ade``/``k_fde`` here, not to ``ade``/``fde``.
        """
        idx, mask, pos = build_token_sequence(data, self.rasterizer, include_future=False)
        B, obs_len, _ = idx.shape
        R = self.cfg.num_samples + 1

        seq = idx.repeat_interleave(R, dim=0)
        key_valid = mask.repeat_interleave(R, dim=0)
        greedy_rows = torch.zeros(B * R, dtype=torch.bool, device=idx.device)
        if self.cfg.greedy_best:
            greedy_rows[::R] = True

        cache = [{} for _ in self.blocks]
        last = self.split_logits(self.logits(seq, key_valid, cache=cache)[:, -1])
        current = seq[:, -1]

        generated = []
        for step in range(self.cfg.pred_len):
            x_logits = self._restrict_to_vicinity(last[0], current[:, 0], self.att_sizes[0])
            y_logits = self._restrict_to_vicinity(last[1], current[:, 1], self.att_sizes[1])
            current = torch.stack([
                self._next_token(x_logits, greedy_rows),
                self._next_token(y_logits, greedy_rows),
                self._next_token(last[2], greedy_rows),
                self._next_token(last[3], greedy_rows),
            ], dim=-1)
            generated.append(current)
            if step == self.cfg.pred_len - 1:
                break
            key_valid = torch.cat([key_valid, torch.ones_like(key_valid[:, :1])], dim=1)
            last = self.split_logits(
                self.logits(current.unsqueeze(1), key_valid, cache=cache, pos_offset=obs_len + step)[:, -1]
            )

        fut = torch.stack(generated, dim=1)
        abs_pos = cells_to_positions(fut[..., 0], fut[..., 1], self.rasterizer)
        abs_pos = abs_pos.view(B, R, self.cfg.pred_len, 2)

        # Hand back relative displacements, so the eval path's cumsum lift applies
        # unchanged and recovers exactly these positions.
        last_obs = pos[:, -1][:, None, None, :].expand(B, R, 1, 2)
        prev = torch.cat([last_obs, abs_pos[:, :, :-1]], dim=2)
        rel = (abs_pos - prev) / POS_SCALE
        return rel[:, 0], rel[:, 1:]
