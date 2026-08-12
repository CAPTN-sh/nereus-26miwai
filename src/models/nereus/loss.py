import torch


def mdn_loss(mdn_out, data, config=None):
    """Negative log likelyhood loss for Nereus
    """
    fut_rel = data.y_rel_pos
    fut_mask = data.y_mask

    B, T, _ = mdn_out.shape
    K = config.mdn_modes

    # unpack MDN
    mdn_out = mdn_out.view(B, T, K, 5)

    pi = mdn_out[..., 0]
    mu = mdn_out[..., 1:3]
    sigma = mdn_out[..., 3:5]

    pi = torch.softmax(pi, dim=-1)
    sigma = torch.exp(sigma) + 1e-6

    # compute loss
    fut_rel = fut_rel.unsqueeze(2)

    dist = torch.distributions.Normal(mu, sigma)
    log_prob = dist.log_prob(fut_rel).sum(dim=-1)

    log_pi = torch.log(pi + 1e-6)
    log_likelihood = torch.logsumexp(log_pi + log_prob, dim=-1)

    # apply mask
    nll = -log_likelihood * fut_mask
    loss = nll.sum() / fut_mask.sum().clamp_min(1)
    loss_series = nll.sum(axis=0) / fut_mask.sum(axis=0).clamp_min(0)

    return loss, {"mdn_nll": loss, "mdn_nll_series": loss_series}
