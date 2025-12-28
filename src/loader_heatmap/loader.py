from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loader_heatmap.trajectories import TrajectoryHeatmapDataset

def loader_heatmap(
    data_folder: Path,
    flag: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    world_size: int,
    rank: int,
    batch_size: int,
    feat_cols=[],
    pin_memory=True,
    normalizer_path = None,
    fut_len = 540,
    obs_len = 120,
):

    file_name = f"{data_folder.parent.name}_{data_folder.name}_{flag}"
    dset = TrajectoryHeatmapDataset(
        nodes_path=data_folder / f"{file_name}_ship_features.parquet",
        min_date=min_date,
        max_date=max_date,
        feat_cols=feat_cols,
        fut_len = fut_len,
        obs_len=obs_len,
    )

    sampler = DistributedSampler(
        dset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=pin_memory,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )
    return dset, sampler, loader


"""
def intent_heatmap_collate_fn(
    samples,
    x_bins: int,
    y_bins: int,
    x_min: float,
    y_min: float,
    pos_res: float,
    sigma_m: float = 75.0,
):
    obs_feat_list, obs_pos_list, obs_mask_list, fut_list = zip(*samples)
    B = len(samples)
    n_cells = x_bins * y_bins

    obs_feat = torch.stack(obs_feat_list, dim=0)   # (B, F, obs_len)
    obs_pos = torch.stack(obs_pos_list, dim=0)     # (B, 2, obs_len)
    obs_mask = torch.stack(obs_mask_list, dim=0)   # (B, obs_len)

    sigma_cells = float(sigma_m) / float(pos_res)
    if sigma_cells <= 0:
        raise ValueError("sigma_m and pos_res must be > 0")

    target = torch.zeros((B, n_cells), dtype=torch.float32)

    for b in range(B):
        fut_pos = fut_list[b]  # torch.Tensor (2, pred_len)
        fut_xy = fut_pos.T.detach().cpu().numpy().astype(np.float32)  # (pred_len, 2)

        grid = np.zeros((x_bins, y_bins), dtype=np.uint8)

        i, j = xy_to_ij(fut_xy, x_bins, y_bins, x_min, y_min, pos_res)

        # OpenCV uses (x, y) == (col, row) so use (j, i)
        pts = np.stack([j, i], axis=1).reshape(-1, 1, 2)

        # draw polyline as 1s
        cv2.polylines(
            grid,
            [pts],
            isClosed=False,
            color=1,
            thickness=1,
            lineType=cv2.LINE_8,
        )

        # TODO guassian

        # Flatten to match your (B, n_cells) target

    return obs_feat, obs_pos, obs_mask, target
"""