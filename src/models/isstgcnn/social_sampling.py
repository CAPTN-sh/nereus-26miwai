"""Social-sampling: negative samples drawn from neighbours' bumper models (IS-STGCNN 3.3).

The paper's idea is that random negative samples "do not reveal anything about
navigational rules", so negatives should instead be placed where a ship must *not* go:
inside the safe-navigation bumper of a nearby ship (Hara, 1991) -- a forward half
ellipse of semi-axes ``6.4 L`` (ahead) by ``1.6 L`` (abeam) closed by an aft half circle
of radius ``1.6 L``, where ``L`` is the ship's length. Points deep inside are "collision"
samples, the rest "risk" samples. The positive sample is the ground-truth position, at a
1:3 positive:negative ratio.

The paper states the sampling strategy and the ratio but never the loss that consumes
them; :func:`models.isstgcnn.loss.social_sampling_loss` supplies that.
"""
import math

import torch

from eval.metrics.cpa import denormalize_static


def _neighbour_heading(y_all, last_obs_pos, x_raw):
    """Heading of every node at every future step, radians clockwise from north.

    Taken from the node's own ground-truth motion, falling back to its last reported
    course over ground when it is barely moving. ``y_all`` [B, N, T, 2],
    ``last_obs_pos`` [B, N, 2], ``x_raw`` [B, N, T_obs, 4] -> [B, N, T].
    """
    prev = torch.cat([last_obs_pos.unsqueeze(2), y_all[:, :, :-1]], dim=2)
    step = y_all - prev
    moving = torch.linalg.norm(step, dim=-1) > 1.0
    cog = torch.deg2rad(x_raw[:, :, -1, 1]).unsqueeze(-1).expand_as(moving)
    return torch.where(moving, torch.atan2(step[..., 0], step[..., 1]), cog)


def sample_bumper_negatives(dense, config, generator=None):
    """Draw ``config.n_negatives`` negative positions per ego and future step.

    Args:
        dense:  the dict from :func:`models.isstgcnn.modules.graph.densify`.
        config: ISSTGCNNParams.

    Returns:
        neg_pos    [B, T, K, 2] absolute positions inside neighbour bumpers,
        neg_weight [B, T, K]    ``collision_weight`` inside the inner band, else 1,
        valid      [B, T]       False where the ego has no neighbour at that step.
    """
    y_all, mask = dense["y_all"], dense["y_all_mask"]
    B, N, T, _ = y_all.shape
    K = config.n_negatives
    device = y_all.device

    if N < 2:
        zeros = y_all.new_zeros(B, T, K, 2)
        return zeros, y_all.new_zeros(B, T, K), torch.zeros(B, T, dtype=torch.bool, device=device)

    # --- pick which neighbour each negative comes from --------------------------------
    nbr_mask = mask[:, 1:].to(y_all.dtype)                       # [B, N-1, T]
    weights = nbr_mask.permute(0, 2, 1).reshape(B * T, N - 1)
    valid = weights.sum(-1) > 0
    # multinomial rejects all-zero rows; give them a dummy distribution and mask later.
    weights = torch.where(valid.unsqueeze(-1), weights, torch.ones_like(weights))
    idx = torch.multinomial(weights, K, replacement=True, generator=generator)  # [B*T, K]
    idx = idx.view(B, T, K) + 1                                  # +1: skip the ego row

    gather_idx = idx.permute(0, 2, 1).reshape(B, K, T, 1)
    pos_j = torch.gather(y_all, 1, gather_idx.expand(B, K, T, 2))  # [B, K, T, 2]

    heading = _neighbour_heading(y_all, dense["pos"][:, :, -1], dense["x_raw"])
    phi = torch.gather(heading, 1, gather_idx.squeeze(-1))         # [B, K, T]

    length = denormalize_static(dense["static"].reshape(B * N, -1))[:, :2].sum(-1)
    length = length.view(B, N).clamp_min(10.0)                     # metres, guard missing dims
    l_j = torch.gather(length.unsqueeze(-1).expand(B, N, T), 1, gather_idx.squeeze(-1))

    # --- uniform point inside the bumper, in ship-local (abeam, ahead) coordinates -----
    a_len = config.bumper_a * l_j
    b_len = config.bumper_b * l_j
    shape = pos_j.shape[:-1]
    rnd = torch.rand(*shape, 3, device=device, generator=generator)
    r = rnd[..., 0].sqrt()
    psi = rnd[..., 1] * math.pi
    # Area-weighted choice between the forward half ellipse and the aft half circle.
    fore = rnd[..., 2] < config.bumper_b / (config.bumper_a + config.bumper_b)
    ahead = torch.where(fore, b_len, a_len) * r * torch.sin(psi) * torch.where(fore, 1.0, -1.0)
    abeam = a_len * r * torch.cos(psi)

    sin_p, cos_p = torch.sin(phi), torch.cos(phi)
    # forward unit vector (sin phi, cos phi); starboard unit vector (cos phi, -sin phi)
    east = pos_j[..., 0] + ahead * sin_p + abeam * cos_p
    north = pos_j[..., 1] + ahead * cos_p - abeam * sin_p
    neg_pos = torch.stack([east, north], dim=-1).permute(0, 2, 1, 3)   # [B, T, K, 2]

    is_collision = r <= config.collision_frac
    weight = torch.where(is_collision, config.collision_weight, 1.0).permute(0, 2, 1)

    return neg_pos, weight, valid.view(B, T)


def sample_random_negatives(dense, config, generator=None):
    """Uniform negatives in a disc around the ego's ground-truth position.

    This is the paper's IS-STGCNN-V2 ablation ("random-sampling ... every position in
    the space has an exact equal chance of being chosen"), which it uses to argue that
    prior-knowledge negatives beat uninformed ones.
    """
    ego_gt = dense["y_all"][:, 0]                                   # [B, T, 2]
    B, T, _ = ego_gt.shape
    K = config.n_negatives

    rnd = torch.rand(B, T, K, 2, device=ego_gt.device, generator=generator)
    r = config.random_radius * rnd[..., 0].sqrt()
    psi = 2.0 * math.pi * rnd[..., 1]
    offset = torch.stack([r * torch.cos(psi), r * torch.sin(psi)], dim=-1)

    weight = torch.ones(B, T, K, device=ego_gt.device, dtype=ego_gt.dtype)
    return ego_gt.unsqueeze(2) + offset, weight, dense["y_all_mask"][:, 0]


def sample_negatives(dense, config, generator=None):
    """Dispatch on ``config.negative_mode``."""
    if config.negative_mode == "random":
        return sample_random_negatives(dense, config, generator)
    if config.negative_mode == "bumper":
        return sample_bumper_negatives(dense, config, generator)
    raise ValueError(f"Unknown negative_mode {config.negative_mode!r}; expected bumper|random")
