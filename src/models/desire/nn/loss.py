import torch


def soft_ce_with_soft_targets(logits, targets):
    """
    logits:  [B,K]
    targets: [B,K], rows sum to 1 (soft labels)
    returns scalar CE = -E_q[log p]
    """
    log_p = torch.log_softmax(logits, dim=1)
    return -(targets * log_p).sum(dim=1).mean()


def k_total_loss(
    pred_sgm,          # [B,K,2,T]   raw SGM samples
    pred_pos_rel_refined,      # [B,K,2,T]   refined = SGM + Δ
    gt,                # [B,2,T]
    mean, logvar,      # [B,L]       CVAE posterior stats
    scores,            # [B,K]       IOC accumulated scores
):
    # 1) Reconstruction (SGM): mean L2 over time & K
    diff_sgm = pred_sgm - gt.unsqueeze(1)           # [B,K,2,T]
    l2_sgm   = torch.norm(diff_sgm, dim=2)          # [B,K,T] (per-step Euclidean)
    l_recon  = l2_sgm.mean()

    # 2) KLD for CVAE posterior vs N(0,I)
    l_kld = (-0.5 * (1 + logvar - mean.pow(2) - logvar.exp())).sum(dim=1).mean()

    # 3) IOC ranking CE with soft targets q = softmax(-max_t L2)
    with torch.no_grad():
        d = torch.norm(pred_sgm - gt.unsqueeze(1), dim=2)     # [B,K,T]
        d_max = d.max(dim=2).values                           # [B,K]
        q = torch.softmax(-d_max, dim=1)                      # [B,K]
    l_rank = soft_ce_with_soft_targets(scores, q)

    # 4) Refinement regression: mean L2 of refined trajectories
    diff_ref = pred_pos_rel_refined - gt.unsqueeze(1)                 # [B,K,2,T]
    l2_ref   = torch.norm(diff_ref, dim=2)                    # [B,K,T]
    l_ref    = l2_ref.mean()

    loss = l_recon + l_kld + l_rank + l_ref
    return loss, (l_recon, l_kld, l_rank, l_ref)