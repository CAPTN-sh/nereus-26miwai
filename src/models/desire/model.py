import torch.nn as nn
import torch

from models.desire.ioc_modules.IOC import IOC
from models.desire.ioc_modules.scene_pooling import ScenePoolingCNN
from models.desire.sgm_modules.SGM import SGM
from models.desire.params import DESIREParams
from data.map.rasterize import Rasterizer

from utils.config import TRAIN_BBOX

class DESIRE(nn.Module):
    def __init__(self, params: DESIREParams):
        super().__init__()
        self.pred_len = params.pred_len
        self.num_refine_iters = params.num_refine_iters

        self.rasterizer = Rasterizer(TRAIN_BBOX, pos_res = 10)

        self.CNN = ScenePoolingCNN(params)
        self.SGM = SGM(params)
        self.IOC = IOC(params, self.rasterizer)
        

    def forward(self, data, scene=None):
        scene_feats = self.CNN(scene).squeeze(0)

        pred_pos_rel, hidde_obs_enc, mean, log_var = self.SGM(data)

        # IOC: scores + per-step Δ for each hypothesis
        iod_params = (pred_pos_rel, hidde_obs_enc, data, scene_feats)
        pred_pos_rel_best, pred_pos_rel_refined, scores = self.IOD_iteration(*iod_params)

        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
        return pred_pos_rel_best, pred_pos_rel[ego_idx], pred_pos_rel_refined, mean, log_var, scores

    def inference(self, data, scene=None):
        scene_feats = self.CNN(scene).squeeze(0)

        # SGM: sample K hypotheses
        pred_pos_rel, hidde_obs_enc, _, _ = self.SGM.inference(data)
        
        # IOC: scores + per-step Δ for each hypothesis
        iod_params = (pred_pos_rel, hidde_obs_enc, data, scene_feats)
        pred_pos_rel_best, pred_pos_rel_refined, scores = self.IOD_iteration(*iod_params)

        return pred_pos_rel_best, pred_pos_rel_refined
    
    def IOD_iteration(self, pred_pos_rel, hidde_obs_enc, data, scene_feats):

        Y = pred_pos_rel
        for _ in range(self.num_refine_iters):
            _, delta = self.IOC(Y, hidde_obs_enc, data, scene_feats)
            Y = Y + delta
        scores, _ = self.IOC(Y, hidde_obs_enc, data, scene_feats)

        best_idx = scores.argmax(dim=1)
        pred_pos_rel_best = Y[torch.arange(Y.size(0), device=Y.device), best_idx]

        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
        return pred_pos_rel_best[ego_idx], Y[ego_idx], scores[ego_idx]