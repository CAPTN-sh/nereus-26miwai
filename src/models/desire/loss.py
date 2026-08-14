import torch


def loss_desire(output, data, config = None):
    pred_pos_rel_best, pred_pos_rel, pred_pos_rel_refined, mean, log_var, scores = output

    fut_pos_rel = data.y_rel_pos.unsqueeze(1)
    fut_mask = data.y_mask
    mask = fut_mask.unsqueeze(1)

    # 1) Reconstruction (SGM)
    l2_sgm = torch.norm(pred_pos_rel - fut_pos_rel, dim=-1) # (per-step Euclidean)
    l_recon = (l2_sgm * mask).sum() / (mask.sum() * pred_pos_rel.shape[1] + 1e-8)

    # 2) KLD for CVAE posterior
    l_kld = (-0.5 * (1 + log_var - mean.pow(2) - log_var.exp())).sum(dim=1)
    l_kld = l_kld.mean()

    # 3) IOC ranking CE with soft targets
    with torch.no_grad():
        d_max = (l2_sgm * mask).max(dim=2).values
        q = torch.softmax(-d_max, dim=1)
    log_p = torch.log_softmax(scores, dim=1)
    l_rank = -(q * log_p).sum(dim=1).mean()

    # 4) Refinement regression
    diff_ref = pred_pos_rel_refined - fut_pos_rel
    l2_ref = torch.norm(diff_ref, dim=-1)
    l_ref = (l2_ref * mask).sum() / (mask.sum() * pred_pos_rel.shape[1] + 1e-8)

    loss = l_recon + l_kld + l_rank + l_ref

    return loss, {"l_recon": l_recon, "l_kld": l_kld, "l_rank": l_rank, "l_ref": l_ref}
