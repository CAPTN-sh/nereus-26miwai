import torch
import torch.nn as nn
import torch.nn.functional as F

from models.desire.nn.social_pooling import SocialPool
from models.desire.utils.get_scene import sample_scene_features
from models.desire.utils.params import DESIREParams

from models.utils.maps.rasterize import Rasterizer


class SCF(nn.Module):
    def __init__(self, params: DESIREParams, rasterizer: Rasterizer):
        super(SCF, self).__init__()
        self.velocity_fc = nn.Linear(params.pred_dim, params.intermediate_size)
        self.social_pool = SocialPool(params)
        self.rasterizer = rasterizer

    def forward_vectorized(self, hidden, pred_pos_abs, pred_pos_rel, seq_start_end, scene_feats):
        pred_pos_abs = pred_pos_abs.permute(0, 2, 1)
        pred_pos_rel = pred_pos_rel.permute(0, 2, 1)

        BK, T, _ = pred_pos_abs.shape

        vel_out = F.relu(self.velocity_fc(pred_pos_rel))
        scene_out = sample_scene_features(
            scene_feats=scene_feats,
            pred_pos_abs=pred_pos_abs.reshape(BK*T, 2),
            rasterizer=self.rasterizer,
            feature_stride=2, # TODO
        ).view(BK, T, -1)

        sp_out = self.social_pool.forward_vectorized(
            pred_pos_abs, hidden, seq_start_end
        )
        return torch.cat((sp_out, vel_out, scene_out), dim=-1)
    

    def forward(self, hidden, pred_pos_abs, pred_pos_rel, seq_start_end, scene_feats):

        vel_out = F.relu(self.velocity_fc(pred_pos_rel))

        scene_out = sample_scene_features(
            scene_feats=scene_feats,
            pred_pos_abs=pred_pos_abs,
            rasterizer=self.rasterizer,
            feature_stride=2, # TODO
        )

        sp_out = self.social_pool(pred_pos_abs, hidden, seq_start_end)
        return torch.cat((sp_out, vel_out, scene_out), 1)
