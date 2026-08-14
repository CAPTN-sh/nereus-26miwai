"""Next-token cross-entropy for the autoregressive TrAISformer."""
import torch
import torch.nn.functional as F

ATTRIBUTES = ("x", "y", "sog", "cog")


def loss_traisformer_ar(output, data, config=None):
    """Sum of the four per-attribute cross-entropies, masked over padding.

    Mirrors the reference implementation: each attribute is scored independently and
    the losses are summed, so a step contributes ``CE(x) + CE(y) + CE(sog) + CE(cog)``.
    ``ce_series`` covers the prediction horizon only, so it lines up with the
    per-timestep series the other models log.

    ``output`` is ``(logits, targets, mask, att_sizes)`` from ``TrAISformerAR.forward``.
    """
    logits, targets, mask, att_sizes = output
    parts = torch.split(logits, list(att_sizes), dim=-1)

    m = mask.float()
    denom = m.sum().clamp_min(1.0)

    per_step = torch.zeros_like(m)
    metrics = {}
    for i, (name, part) in enumerate(zip(ATTRIBUTES, parts)):
        ce = F.cross_entropy(
            part.reshape(-1, part.size(-1)), targets[..., i].reshape(-1), reduction="none"
        ).view_as(m)
        per_step = per_step + ce
        metrics[f"ce_{name}"] = ((ce * m).sum() / denom).detach()

    loss = (per_step * m).sum() / denom

    pred_len = getattr(config, "pred_len", 0) or 0
    tail, tail_m = per_step[:, -pred_len:], m[:, -pred_len:]
    series = (tail * tail_m).sum(dim=0) / tail_m.sum(dim=0).clamp_min(1.0)

    with torch.no_grad():
        for i, name in enumerate(ATTRIBUTES[:2]):
            hit = (parts[i].argmax(-1) == targets[..., i]).float()
            metrics[f"acc_{name}"] = (hit * m).sum() / denom

    return loss, {"ce": loss.detach(), "ce_series": series.detach(), **metrics}
