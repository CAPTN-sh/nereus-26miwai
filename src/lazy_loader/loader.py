from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.lazy_loader.trajectories import LazyTrajectoryDataset, seq_collate


def lazy_loader(
    data_folder: Path,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    world_size: int,
    rank: int,
    batch_size: int,
    feat_cols=[],
):

    dset = LazyTrajectoryDataset(
        nodes_path=data_folder / "nodes.parquet",
        edges_path=data_folder / "edges.parquet",
        normalizer_path=data_folder / "normalization_stats.npy",
        min_date=min_date,
        max_date=max_date,
        feat_cols=feat_cols,
    )

    sampler = DistributedSampler(dset, num_replicas=world_size, rank=rank, shuffle=True)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=seq_collate,
        pin_memory=True,
        drop_last=True,
    )
    return dset, sampler, loader
