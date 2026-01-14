import torch
import torch.nn.functional as F
from torch import amp
from models.utils.maps.rasterize import Rasterizer

def sample_scene_features(scene_feats, pred_pos_abs, rasterizer: Rasterizer, feature_stride):
    """
    rho_I:         [C, Hf, Wf] scene CNN feature map ρ(I)
    y_world_m:     [B, 2] agent positions (meters, same UTM as your BEV)
    world_to_bev:  3x3 numpy/torch array mapping [x_m,y_m,1] -> [u_px,v_px,1] in BEV pixels
    feature_stride:int total downsample from BEV -> ρ(I) (e.g., 2 if conv1 stride=2)
    returns:       [B, C] pooled features at each agent location (bilinear)
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