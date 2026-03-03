import torch
import torch.nn as nn
import torch.nn.functional as F

from models.desire.ioc_modules.social_pooling import SocialPool
from models.desire.ioc_modules.get_scene import sample_scene_features
from models.desire.params import DESIREParams
from data.map.rasterize import Rasterizer


class SCF(nn.Module):
    """
    Scene Context Fusion (SCF):
     - map context through scene sampling
     - interaction context through social pooling
    """
    def __init__(self, params: DESIREParams, rasterizer: Rasterizer):
        super(SCF, self).__init__()
        self.velocity_fc = nn.Linear(params.pred_dim, params.intermediate_size)
        self.social_pool = SocialPool(params)
        self.rasterizer = rasterizer
    
    def forward(self, hidden, pred_pos_abs, pred_pos_rel, data, scene_feats):
        N, K, T, H = hidden.shape
        vel_out = F.relu(self.velocity_fc(pred_pos_rel))

        scene_out = sample_scene_features(
            scene_feats=scene_feats,
            pred_pos_abs=pred_pos_abs.reshape(N * K * T, 2),
            rasterizer=self.rasterizer,
            feature_stride=2, # TODO
        ).view(N, K, T, -1)

        sp_out = self.social_pool(pred_pos_abs, hidden, data)

        return torch.cat((sp_out, vel_out, scene_out), dim=-1)
