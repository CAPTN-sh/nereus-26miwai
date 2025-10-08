import torch.nn as nn

from models.desire.IOC import IOC
from models.desire.nn.scene_pooling import ScenePoolingCNN
from models.desire.SGM import SGM
from models.desire.utils.params import DESIREParams


class DESIRE(nn.Module):
    def __init__(self, params: DESIREParams):
        super().__init__()
        self.pred_len = params.pred_len
        self.num_refine_iters = params.num_refine_iters

        self.CNN = ScenePoolingCNN(params)
        self.SGM = SGM(params)
        self.IOC = IOC(params)

    def forward(self, batch, scene, scene_meta):
        obs_feat, obs_pos, obs_pos_rel, fut_pos, fut_pos_rel, seq_start_end = batch
        obs_pos_last = obs_pos[:, :, -1]

        scene_feats = self.CNN(scene).squeeze(0)

        # SGM: sample K hypotheses
        #use_prior = (torch.rand((), device=obs_pos.device) < 0.5)
        if False:
            pred_pos_rel, hidde_obs_enc, mean, log_var = self.SGM.inference(obs_pos_rel)
        else:
            pred_pos_rel, hidde_obs_enc, mean, log_var = self.SGM(obs_pos_rel, fut_pos_rel)

        # IOC: scores + per-step Δ for each hypothesis
        pred_pos_rel_best, pred_pos_rel_refined, scores = self.IOD_iteration(
            self.num_refine_iters,
            pred_pos_rel,
            hidde_obs_enc,
            obs_pos_last,
            seq_start_end,
            scene_feats,
            scene_meta
        )

        return pred_pos_rel_best, pred_pos_rel, pred_pos_rel_refined, mean, log_var, scores

    def inference(self, batch, scene, scene_meta):
        obs_feat, obs_pos, obs_pos_rel, fut_pos, fut_pos_rel, seq_start_end = batch
        obs_pos_last = obs_pos[:, :, -1]

        scene_feats = self.CNN(scene).squeeze(0)

        # SGM: sample K hypotheses
        pred_pos_rel, hidde_obs_enc, mean, log_var = self.SGM.inference(obs_pos_rel)
        
        # IOC: scores + per-step Δ for each hypothesis
        pred_pos_rel_best, pred_pos_rel_refined, scores = self.IOD_iteration(
            self.num_refine_iters, 
            pred_pos_rel, 
            hidde_obs_enc, 
            obs_pos_last, 
            seq_start_end, 
            scene_feats, 
            scene_meta
        )

        return pred_pos_rel_best, pred_pos_rel_refined
    
    def IOD_iteration(self, num_refine_iters, pred_pos_rel, hidde_obs_enc, obs_pos_last, seq_start_end, scene_feats, scene_meta):
        scores, delta = self.IOC(
            pred_pos_rel, hidde_obs_enc, obs_pos_last, seq_start_end, scene_feats, scene_meta
        )
        pred_pos_rel_refined = pred_pos_rel + delta

        for _ in range(num_refine_iters):
            scores, delta = self.IOC(
                pred_pos_rel_refined,
                hidde_obs_enc,
                obs_pos_last,
                seq_start_end,
                scene_feats,
                scene_meta,
            )
            pred_pos_rel_refined = pred_pos_rel_refined + delta

        scores, _ = self.IOC(
            pred_pos_rel_refined,
            hidde_obs_enc,
            obs_pos_last,
            seq_start_end,
            scene_feats,
            scene_meta,
        )

        best_idx = scores.argmax(dim=1)
        pred_pos_rel_best = pred_pos_rel_refined.gather(
            1, best_idx.view(-1, 1, 1, 1).expand(-1, 1, 2, self.pred_len)
        ).squeeze(1)

        return pred_pos_rel_best, pred_pos_rel_refined, scores
