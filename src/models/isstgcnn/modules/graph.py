"""Dense spatio-temporal graph construction for IS-STGCNN.

The dataloader hands out a *directed star* ``edge_index`` (neighbour -> ego, one
timestep) which carries no neighbour-to-neighbour structure, so it cannot be used as
the ST-GCNN kernel. This module rebuilds what Social-STGCNN needs instead: a dense,
per-timestep, symmetric adjacency over the ego and its neighbours, from ``x_pos``.
"""
import torch
from torch_geometric.utils import to_dense_batch

# Positions are metres; distances between ships are O(100 m) while Social-STGCNN was
# tuned on pedestrians O(1 m) apart. Without rescaling, 1/d is ~0.003 and the added
# self-loop swamps it, leaving A_hat ~= I and no social signal at all. Dividing by
# DIST_SCALE puts inter-ship distances back in the O(1) regime the kernel expects.
DIST_SCALE = 100.0

EPS = 1e-6
MAX_EDGE_WEIGHT = 1e3


def densify(data, max_nodes: int | None = None) -> dict:
    """Scatter the flat PyG batch into dense ``[B, N, ...]`` tensors.

    Node order within a graph is preserved by ``to_dense_batch``, and the dataset emits
    the ego first followed by neighbours sorted by distance, so truncating to
    ``max_nodes`` keeps the ego plus its nearest neighbours.

    Returns a dict with:
        v          [B, N, T_obs, 2]  relative displacement (already /100)
        pos        [B, N, T_obs, 2]  absolute position in metres
        step_mask  [B, N, T_obs]     bool, False on the left pad
        node_mask  [B, N]            bool, False on the dense pad
        static     [B, N, 8]
        x_raw      [B, N, T_obs, 4]  raw speed / course / acc / angular difference
        y_all      [B, N, T_pred, 2] absolute future positions of every node
        y_all_mask [B, N, T_pred]    bool
    """
    batch = data.batch
    kwargs = {"max_num_nodes": max_nodes} if max_nodes is not None else {}

    v, node_mask = to_dense_batch(data.x[..., :2], batch, **kwargs)
    pos, _ = to_dense_batch(data.x_pos, batch, **kwargs)
    step_mask, _ = to_dense_batch(data.x_mask, batch, **kwargs)
    static, _ = to_dense_batch(data.static, batch, **kwargs)
    x_raw, _ = to_dense_batch(data.x_raw, batch, **kwargs)
    y_all, _ = to_dense_batch(data.y_all.squeeze(1), batch, **kwargs)
    y_all_mask, _ = to_dense_batch(data.y_all_mask, batch, **kwargs)
    is_ego, _ = to_dense_batch(data.is_ego, batch, **kwargs)

    return {
        "v": v,
        "is_ego": is_ego & node_mask,
        "pos": pos,
        "step_mask": step_mask & node_mask.unsqueeze(-1),
        "node_mask": node_mask,
        "static": static,
        "x_raw": x_raw,
        "y_all": y_all,
        "y_all_mask": y_all_mask.bool() & node_mask.unsqueeze(-1),
    }


def _diag_variance(pos, valid):
    """Per-graph diagonal coordinate covariance over the valid (node, step) positions.

    ``pos`` [B, N, T, 2], ``valid`` [B, N, T] -> [B, 2], floored so single-ship or
    stationary scenes cannot produce a singular S.
    """
    w = valid.unsqueeze(-1).to(pos.dtype)
    n = w.sum(dim=(1, 2)).clamp_min(1.0)
    mean = (pos * w).sum(dim=(1, 2)) / n
    var = (((pos - mean[:, None, None, :]) ** 2) * w).sum(dim=(1, 2)) / n
    return var.clamp_min(1.0)  # metres^2


def build_adjacency(pos, node_mask, step_mask, kernel: str = "mahalanobis"):
    """Per-timestep adjacency kernel. ``pos`` [B, N, T, 2] -> A [B, T, N, N].

    ``euclid``
        Social-STGCNN's original ``a_ij = 1/||p_i - p_j||`` (Mohamed et al. 2020),
        on positions rescaled by :data:`DIST_SCALE`.
    ``mahalanobis``
        IS-STGCNN Eq. 3: ``a_ij = 1/sqrt((p_i-p_j)^T S^-1 (p_i-p_j))`` with S the
        diagonal coordinate covariance of the scene. Unit-free, so it needs no
        rescaling -- which is a large part of why the paper prefers it.
    ``eq4_literal``
        IS-STGCNN Eq. 4 exactly as printed: ``1/sqrt((dx)^2 * sigma * (dy)^2)``, zero
        when dx or dy is 0. Note this is a *product*, not a sum: it is dimensionally
        inconsistent and diverges as either component goes to zero (i.e. for ships on
        the same latitude or longitude), which is why it is not the default. Kept so
        the printed formula can be compared against the stated intent of Eq. 3.
    """
    valid = step_mask & node_mask.unsqueeze(-1)              # [B, N, T]
    p = pos.permute(0, 2, 1, 3)                              # [B, T, N, 2]
    diff = p.unsqueeze(-2) - p.unsqueeze(-3)                 # [B, T, N, N, 2] = p_i - p_j

    if kernel == "mahalanobis":
        var = _diag_variance(pos, valid)                     # [B, 2]
        d = torch.sqrt((diff ** 2 / var[:, None, None, None, :]).sum(-1) + EPS)
        a = torch.where(d > EPS, 1.0 / d, torch.zeros_like(d))
    elif kernel == "euclid":
        d = torch.linalg.norm(diff / DIST_SCALE, dim=-1)
        a = torch.where(d > EPS, 1.0 / d.clamp_min(EPS), torch.zeros_like(d))
    elif kernel == "eq4_literal":
        # Eq. 5: sigma is the spread of the coordinates about their mean.
        sigma = _diag_variance(pos, valid).sum(-1).sqrt()    # [B]
        d2 = diff[..., 0] ** 2 * sigma[:, None, None, None] * diff[..., 1] ** 2
        nonzero = (diff[..., 0].abs() > EPS) & (diff[..., 1].abs() > EPS)
        a = torch.where(nonzero, 1.0 / d2.clamp_min(EPS).sqrt(), torch.zeros_like(d2))
    else:
        raise ValueError(f"Unknown adj_kernel {kernel!r}; expected euclid|mahalanobis|eq4_literal")

    # Duplicate AIS fixes put two ships at the same coordinates, where 1/d explodes and
    # would swamp every other edge after normalisation.
    a = a.clamp_max(MAX_EDGE_WEIGHT)

    pair_valid = valid.permute(0, 2, 1)                      # [B, T, N]
    pair_valid = pair_valid.unsqueeze(-1) & pair_valid.unsqueeze(-2)
    a = a * pair_valid.to(a.dtype)
    return a * (1.0 - torch.eye(a.size(-1), device=a.device, dtype=a.dtype))


def normalize_adj(a, node_mask):
    """Symmetric normalisation ``A_hat = D^-1/2 (A + I) D^-1/2``, per timestep.

    Self-loops (and hence non-zero rows) are added only for real nodes, so padded
    rows/columns stay exactly zero.
    """
    valid = node_mask.to(a.dtype)                            # [B, N]
    eye = torch.eye(a.size(-1), device=a.device, dtype=a.dtype)
    a = a + eye * valid[:, None, None, :] * valid[:, None, :, None]
    deg_inv_sqrt = a.sum(-1).clamp_min(EPS).pow(-0.5) * valid[:, None, :]
    return deg_inv_sqrt.unsqueeze(-1) * a * deg_inv_sqrt.unsqueeze(-2)
