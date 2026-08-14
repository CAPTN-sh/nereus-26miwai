import torch
import torch.nn.functional as F
from torch import amp

from data.map.rasterize import Rasterizer


def sample_scene_features(scene_feats, pred_pos_abs, rasterizer: Rasterizer, feature_stride):
    """Samples the scene features from the cnn at a given predicted abs position.
    For this the position is maped onto the cnn grid.
    """
    device = scene_feats.device
    feat_dtype = scene_feats.dtype
    C, Hf, Wf = scene_feats.shape

    with amp.autocast("cuda", enabled=False):
        pred = pred_pos_abs.to(device=device, dtype=torch.float32)
        B = pred.shape[0]

        # BEV world(m) -> pixels -> feature pixels
        u = (pred[:, 0] - rasterizer.x_min) / rasterizer.pos_res / float(feature_stride)
        v = (pred[:, 1] - rasterizer.y_min) / rasterizer.pos_res / float(feature_stride)

        # normalize to [-1,1] for grid_sample
        Wn = max(Wf - 1, 1)
        Hn = max(Hf - 1, 1)
        x_norm = 2.0 * (u / Wn) - 1.0
        y_norm = 2.0 * (v / Hn) - 1.0

        grid = torch.stack([x_norm, y_norm], dim=1).view(1, 1, B, 2)  # [1,1,B,2]

    grid = grid.to(dtype=feat_dtype)

    # grid_sample wants [N,C,H,W], so add batch dim
    feat = F.grid_sample(
        scene_feats.unsqueeze(0), grid, align_corners=True, padding_mode="zeros"
    )
    feat = feat.squeeze(0).squeeze(1).transpose(0, 1).contiguous()
    return feat
