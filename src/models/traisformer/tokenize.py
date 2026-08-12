"""Turn an AIS batch into the ``(x, y, SOG, COG)`` token sequence TrAISformer consumes.

The original model tokenises four attributes per timestep and predicts the next token.
This repo stores the observation window and the ego future separately, so the sequence
is stitched from ``x_pos``/``x_raw`` (observed) and ``y_pos`` (future).

The future has no reported SOG/COG, so both are derived from consecutive positions.
That is sound here: measured on real batches, the derived course matches the reported
COG to ~1 degree median, and ``|dp| / speed`` comes out at 9.99 -- i.e. the stored speed
is metres per second over a ``STEP_SIZE`` grid, not knots.
"""
import torch

from utils.config import STEP_SIZE


def kinematics_from_positions(pos, prev):
    """Speed [m/s] and course [deg from north] implied by ``pos - prev``."""
    step = pos - prev
    speed = torch.linalg.norm(step, dim=-1) / STEP_SIZE
    course = torch.rad2deg(torch.atan2(step[..., 0], step[..., 1])) % 360.0
    return speed, course


def build_token_sequence(data, rasterizer, include_future: bool = True):
    """Discretise the ego trajectory into token indices.

    Returns:
        idx  [B, L, 4] int64 -- (x_cell, y_cell, sog_bin, cog_bin)
        mask [B, L]    bool  -- False on left padding and beyond the trajectory end
        pos  [B, L, 2] float -- the absolute positions the tokens came from
    where ``L = obs_len (+ pred_len)``.
    """
    ego = data.is_ego.nonzero(as_tuple=True)[0]
    obs_pos, obs_raw, obs_mask = data.x_pos[ego], data.x_raw[ego], data.x_mask[ego]

    pos, mask = obs_pos, obs_mask
    sog, cog = obs_raw[..., 0], obs_raw[..., 1]

    if include_future:
        fut_pos = data.y_pos
        prev = torch.cat([obs_pos[:, -1:], fut_pos[:, :-1]], dim=1)
        fut_sog, fut_cog = kinematics_from_positions(fut_pos, prev)
        pos = torch.cat([obs_pos, fut_pos], dim=1)
        mask = torch.cat([obs_mask, data.y_mask], dim=1)
        sog = torch.cat([sog, fut_sog], dim=1)
        cog = torch.cat([cog, fut_cog], dim=1)

    x_idx, y_idx = rasterizer.pos_to_index(pos)
    sog_idx, cog_idx = rasterizer.kin_to_index(torch.stack([sog, cog], dim=-1))
    return torch.stack([x_idx, y_idx, sog_idx, cog_idx], dim=-1), mask, pos


def cells_to_positions(x_idx, y_idx, rasterizer):
    """Inverse of ``pos_to_index``: cell indices -> metric cell centres."""
    x = (x_idx.float() + 0.5) * rasterizer.pos_res + rasterizer.x_min
    y = (y_idx.float() + 0.5) * rasterizer.pos_res + rasterizer.y_min
    return torch.stack([x, y], dim=-1)
