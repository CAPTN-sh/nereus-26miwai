"""Model-predictive correction of a predicted trajectory (IS-STGCNN Eq. 6-9).

The network output can violate ship dynamics, so the paper re-solves for the closest
trajectory a ship could actually steer, under the rudder constraint |delta| <= 35 deg.
The paper's prediction model is a simplified MMG; we use first-order Nomoto steering,

    r_dot = (K * delta - r) / T,   phi_dot = r,
    x_dot = u * sin(phi),          y_dot = u * cos(phi),

with x east / y north and phi measured clockwise from north (the same convention as
``eval/metrics/cpa.rotate_rectangle``). The speed profile u_t is taken from the
predicted trajectory itself, so the correction only reshapes *heading* into something a
rudder can produce and leaves the along-track speed alone.

Eq. 6 is minimised directly by gradient descent on the rudder sequence: the rollout is
differentiable and the constraint is enforced by construction via ``delta_max * tanh``.
"""
import torch

from utils.config import STEP_SIZE


def _rollout(delta, phi0, r0, p0, speed, nomoto_k, nomoto_t, dt):
    """Integrate the Nomoto model. Returns positions [B, T, 2]."""
    phi, r, p = phi0, r0, p0
    out = []
    for t in range(delta.size(1)):
        r = r + dt * (nomoto_k * delta[:, t] - r) / nomoto_t
        phi = phi + dt * r
        step = speed[:, t] * dt
        p = p + torch.stack([step * torch.sin(phi), step * torch.cos(phi)], dim=-1)
        out.append(p)
    return torch.stack(out, dim=1)


def mpc_correct(pred_abs, last_pos, course_deg, ang_diff_deg, config):
    """Return the kinematically feasible trajectory closest to ``pred_abs``.

    Args:
        pred_abs:      [B, T, 2] predicted absolute positions (metres).
        last_pos:      [B, 2] last observed position.
        course_deg:    [B] last observed course over ground, degrees from north.
        ang_diff_deg:  [B] last observed course change per step, degrees.
        config:        ISSTGCNNParams (nomoto_k/nomoto_t/rudder_max_deg/mpc_iters/mpc_lr).

    Returns:
        [B, T, 2] corrected absolute positions.
    """
    # Eval runs inside ``inference_mode``; clone out of it so autograd can be used.
    with torch.inference_mode(False), torch.enable_grad():
        pred = pred_abs.clone().detach()
        p0 = last_pos.clone().detach()
        dt = float(STEP_SIZE)

        prev = torch.cat([p0.unsqueeze(1), pred[:, :-1]], dim=1)
        speed = torch.linalg.norm(pred - prev, dim=-1) / dt          # [B, T] m/s

        # Heading: prefer the observed direction of travel, fall back to reported COG
        # when the ship is effectively stationary and the displacement is just noise.
        obs_step = pred[:, 0] - p0
        moving = torch.linalg.norm(obs_step, dim=-1) > 1.0
        cog = torch.deg2rad(course_deg.clone().detach())
        phi0 = torch.where(moving, torch.atan2(obs_step[..., 0], obs_step[..., 1]), cog)
        r0 = torch.deg2rad(ang_diff_deg.clone().detach()) / dt

        delta_max = torch.deg2rad(torch.tensor(config.rudder_max_deg, device=pred.device))
        theta = torch.zeros_like(speed, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=config.mpc_lr)

        for _ in range(config.mpc_iters):
            opt.zero_grad(set_to_none=True)
            roll = _rollout(delta_max * torch.tanh(theta), phi0, r0, p0, speed,
                            config.nomoto_k, config.nomoto_t, dt)
            loss = ((roll - pred) ** 2).sum(-1).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            return _rollout(delta_max * torch.tanh(theta), phi0, r0, p0, speed,
                            config.nomoto_k, config.nomoto_t, dt).detach()
