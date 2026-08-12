"""Losses for IS-STGCNN: bivariate Gaussian NLL plus the social-sampling term."""
import math

import torch

from models.isstgcnn.social_sampling import sample_negatives

POS_SCALE = 100.0   # repo-wide normalisation of relative displacements
LOG_SIGMA_RANGE = (-7.0, 5.0)
EPS = 1e-6


def bivariate_logp(params, target):
    """Log density of the bivariate Gaussian parameterised by ``params``.

    ``params[..., :5]`` = (mu_x, mu_y, log_sigma_x, log_sigma_y, atanh(rho)) as in
    Social-STGCNN; ``target[..., :2]`` is a relative displacement in the same /100 units.
    """
    mu = params[..., 0:2]
    log_sigma = params[..., 2:4].clamp(*LOG_SIGMA_RANGE)
    sigma = torch.exp(log_sigma)
    rho = torch.tanh(params[..., 4]).clamp(-0.99, 0.99)

    norm = (target - mu) / sigma
    nx, ny = norm[..., 0], norm[..., 1]
    one_minus_rho2 = (1.0 - rho ** 2).clamp_min(EPS)
    z = nx ** 2 + ny ** 2 - 2.0 * rho * nx * ny
    return (
        -z / (2.0 * one_minus_rho2)
        - math.log(2.0 * math.pi)
        - log_sigma.sum(-1)
        - 0.5 * torch.log(one_minus_rho2)
    )


def rel_targets(dense, data=None):
    """Per-step relative displacement targets for *every* node, from absolute ``y_all``.

    ``y_all`` holds absolute future positions, so the displacement at step 0 is measured
    against the last observed position and at step t against step t-1.

    Absolute positions are UTM metres stored as float32, where one ulp is already ~0.5 m,
    so differencing them costs about that much precision (the dataset itself differences
    in float64 before casting). Neighbour targets carry that noise -- negligible against
    the ~50-100 m a ship moves per step -- but the ego's exact target is available as
    ``y_rel_pos``, so it is substituted in when ``data`` is given.
    """
    prev = torch.cat([dense["pos"][:, :, -1:], dense["y_all"][:, :, :-1]], dim=2)
    target = (dense["y_all"] - prev) / POS_SCALE
    if data is not None:
        target = target.masked_scatter(dense["is_ego"][..., None, None], data.y_rel_pos)
    return target


def social_sampling_loss(params_ego, target_ego, mask_ego, dense, config):
    """InfoNCE between the ground-truth displacement and bumper-model negatives.

    The paper specifies the sampling strategy (positives from ground truth, negatives
    inside neighbours' bumpers, ratio 1:3) but not the objective. We contrast them under
    the model's own predictive density, which is what "steer clear of any areas that
    could result in a collision or risk" amounts to for a probabilistic decoder: mass is
    pushed off positions inside a neighbour's safety bumper and onto the ground truth.
    """
    neg_pos, weight, has_nbr = sample_negatives(dense, config)         # [B,T,K,·]

    ego_prev = torch.cat([dense["pos"][:, 0, -1:], dense["y_all"][:, 0, :-1]], dim=1)
    d_neg = (neg_pos - ego_prev.unsqueeze(2)) / POS_SCALE               # [B, T, K, 2]

    lp_pos = bivariate_logp(params_ego, target_ego)                     # [B, T]
    lp_neg = bivariate_logp(params_ego.unsqueeze(2), d_neg)             # [B, T, K]

    logits = torch.cat([lp_pos.unsqueeze(-1), lp_neg + torch.log(weight.clamp_min(EPS))], dim=-1)
    nce = -(lp_pos - torch.logsumexp(logits, dim=-1))

    mask = (mask_ego & has_nbr).to(nce.dtype)
    return (nce * mask).sum() / mask.sum().clamp_min(1.0)


def loss_isstgcnn(output, data, config=None):
    """Masked bivariate NLL (+ optional social-sampling term).

    ``output`` is ``(params [B, N, T_pred, 5], dense)`` as returned by ``ISSTGCNN.forward``.
    Supervision covers every node in the scene when ``config.supervise == "all"`` (the
    Social-STGCNN convention), or the ego only otherwise.
    """
    params_all, dense = output
    target_all = rel_targets(dense, data)
    mask_all = dense["y_all_mask"]

    if config.supervise == "ego":
        params, target, mask = params_all[:, :1], target_all[:, :1], mask_all[:, :1]
    else:
        params, target, mask = params_all, target_all, mask_all

    m = mask.to(params.dtype)
    nll = -bivariate_logp(params, target) * m
    denom = m.sum().clamp_min(1.0)
    loss = nll.sum() / denom
    nll_series = nll.sum(dim=(0, 1)) / m.sum(dim=(0, 1)).clamp_min(1.0)

    parts = {"nll": loss.detach(), "nll_series": nll_series.detach()}
    if config.social_sampling:
        social = social_sampling_loss(
            params_all[:, 0], target_all[:, 0], mask_all[:, 0], dense, config
        )
        parts["social"] = social.detach()
        loss = loss + config.lambda_social * social

    return loss, parts
