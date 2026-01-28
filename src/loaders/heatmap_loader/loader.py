from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loaders.heatmap_loader.trajectories import TrajectoryHeatmapDataset

from utils.config import AIS_SOURCE

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
    fut_len = 540,
    obs_len = 120,
):

    file_name = f"{AIS_SOURCE}_{data_folder.name}_{flag}"
    dset = TrajectoryHeatmapDataset(
        nodes_path=data_folder / f"{file_name}_ship_features.parquet",
        flag=flag,
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