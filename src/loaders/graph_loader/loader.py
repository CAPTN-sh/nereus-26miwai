from pathlib import Path

import pandas as pd

from torch_geometric.loader import DataLoader
from loaders.graph_loader.trajectories import GraphTrajectoryDataset

from utils.config import AIS_SOURCE

def graph_loader(
    data_folder: Path,
    flag: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    batch_size: int,
    feat_cols=[],
    pin_memory=True,
    pred_len=30,
    obs_len=60,
):

    file_name = f"{AIS_SOURCE}_{data_folder.name}_{flag}"
    dset = GraphTrajectoryDataset(
        nodes_path=data_folder / f"{file_name}_ship_features.parquet",
        edges_path=data_folder / f"{file_name}_ship2ship_features.parquet",
        flag=flag,
        min_date=min_date,
        max_date=max_date,
        feat_cols=feat_cols,
        pred_len=pred_len,
        obs_len=obs_len,
    )

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=False,
        drop_last=True,
    )
    return loader
