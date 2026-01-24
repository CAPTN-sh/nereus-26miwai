from pathlib import Path
import pandas as pd
from scene_loader.loader import SceneTrajectoryDataset

from utils.config import DATA_FOLDER_PATH

feat_cols = ["speed", "course", "acc", "angular_difference", "length", "width", "ship_group"]
data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"
for l in ["val", "train"]: # "train", 
    file_name = f"{data_folder.parent.name}_{data_folder.name}_{l}"
    dset = SceneTrajectoryDataset(
        nodes_path=data_folder / f"{file_name}_ship_features.parquet",
        edges_path=data_folder / f"{file_name}_ship2ship_features.parquet",
        flag=l,
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        feat_cols=feat_cols,
        force_rebuild=True,
    )