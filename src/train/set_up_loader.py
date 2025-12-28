from pathlib import Path
import pandas as pd
from sceen_loader.loader import sceen_loader

data_folder = Path("/home/bbiesenbach/ais.processing/data/ais/4_features")
train_dset, train_sampler, train_loader = sceen_loader(
    data_folder=data_folder / "fhkiel_train/kiel",
    min_date=pd.Timestamp("2022-01-01"),
    max_date=pd.Timestamp("2024-01-01"), # hyper: 2022-05-16, full: 2024-01-01
    world_size=1,
    rank=0,
    batch_size=1024 * 8,
    pin_memory=False,
    feat_cols=["speed", "course"],
    normalizer_path = None
)

train_dset, train_sampler, train_loader = sceen_loader(
    data_folder=data_folder / "fhkiel_train/kiel",
    min_date=pd.Timestamp("2022-01-01"),
    max_date=pd.Timestamp("2024-01-01"), # hyper: 2023-03-25, full: 2024-01-01 
    world_size=1,
    rank=0,
    batch_size=1024 * 8,
    pin_memory=False,
    feat_cols=["speed", "course"],
    normalizer_path = "/home/bbiesenbach/shipwise/data/cache/sceen_loader_2022-01-01_2024-01-01_24_24__sc_fhkiel_trainkiel_normalizer.pkl"
)