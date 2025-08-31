import torch
import torch.nn as nn
import torch.nn.functional as F

from models.desire.nn.social_pooling import SocialPool
from models.desire.utils.get_scene import sample_scene_features
from models.desire.utils.params import DESIREParams


class SCF(nn.Module):
    def __init__(self, params: DESIREParams):
        super(SCF, self).__init__()
        self.velocity_fc = nn.Linear(params.pred_dim, params.intermediate_size)
        self.social_pool = SocialPool(params)

    def forward(
        self, hidden, pred_pos_abs, pred_pos_rel, seq_start_end, scene_feats, scene_meta
    ):
        pos_to_px = scene_meta["world_to_bev"]
        feature_stride = max(
            scene_meta["size_px"][0] // scene_feats.shape[-1],
            scene_meta["size_px"][1] // scene_feats.shape[-2],
        )

        scene_out = sample_scene_features(
            scene_feats=scene_feats,
            pred_pos_abs=pred_pos_abs,
            pos_to_px=pos_to_px,
            feature_stride=feature_stride,
        )

        vel_out = F.relu(self.velocity_fc(pred_pos_rel))
        sp_out = self.social_pool(pred_pos_abs, hidden, seq_start_end)
        return torch.cat((sp_out, vel_out, scene_out), 1)
