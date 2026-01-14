import torch

def loss_desire(output, batch, epoch, config):
    pred_pos_rel_best, pred_pos_rel, pred_pos_rel_refined, mean, log_var, scores = output
    *_, fut_pos_rel, fut_mask, _ = batch

    mask = fut_mask.squeeze(1).unsqueeze(1)
    agent_valid = (mask.sum(dim=2) > 0)
    agent_valid_f = agent_valid.float()

    # 1) Reconstruction (SGM): mean L2 over time & K
    l2_sgm = torch.norm(pred_pos_rel - fut_pos_rel.unsqueeze(1), dim=2) # [B,K,T] (per-step Euclidean)
    l_recon = (l2_sgm * mask).sum() / (mask.sum() * l2_sgm.shape[1] + 1e-8)

    # 2) KLD for CVAE posterior vs N(0,I)
    l_kld = (-0.5 * (1 + log_var - mean.pow(2) - log_var.exp())).sum(dim=1)
    l_kld = (l_kld * agent_valid_f.squeeze(1)).sum() / (agent_valid_f.sum() + 1e-8)

    # 3) IOC ranking CE with soft targets q = softmax(-max_t L2)
    with torch.no_grad():
        d_max = (l2_sgm.masked_fill(mask == 0, -1e9)).max(dim=2).values
        d_max = d_max.masked_fill(~agent_valid, 0.0)
        q = torch.softmax(-d_max, dim=1)  # [B,K]
    log_p = torch.log_softmax(scores, dim=1)
    ce = -(q * log_p).sum(dim=1)
    l_rank = (ce * agent_valid_f.float()).sum() / (agent_valid_f.sum() + 1e-8)

    # 4) Refinement regression: mean L2 of refined trajectories
    diff_ref = pred_pos_rel_refined - fut_pos_rel.unsqueeze(1)  # [B,K,2,T]
    l2_ref = torch.norm(diff_ref, dim=2)  # [B,K,T]
    l_ref = (l2_ref * mask).sum() / (mask.sum() * l2_sgm.shape[1] + 1e-8)

    beta = min(0.5, (epoch + 1) / 8)
    loss = l_recon + beta * l_kld + l_rank + l_ref

    return loss, {"l_recon": l_recon, "l_kld": l_kld, "l_rank": l_rank, "l_ref": l_ref}
