from utils.config import Config
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

config = Config("src/preprocessing/configs/_main.yaml")
out_folder = Path(config.get("decode")["paths"]["out_folder"])

paths = list(out_folder.glob("*_ship.parquet"))
dfs = [
    pd.read_parquet(path, engine="pyarrow")
    for path in tqdm(paths, desc="Loading ship data")
]
df_ship = pd.concat(dfs).reset_index(drop=True)

paths = list(out_folder.glob("*_traj.parquet"))
dfs = [
    pd.read_parquet(path, engine="pyarrow")
    for path in tqdm(paths, desc="Loading traj data")
]
df_traj = pd.concat(dfs).reset_index(drop=True)

columns = ["ship_type", "to_bow", "to_stern", "to_port", "to_starboard"]
df_summary = df_ship.groupby("mmsi")[columns].max()

missing_mmsis = set(df_traj["mmsi"].unique()) - set(df_summary.index)
missing_df = pd.DataFrame(data=np.nan, index=list(missing_mmsis), columns=columns)
missing_df.index.name = "mmsi"

for col in columns:
    missing_df[col] = missing_df[col].astype("Int64")

df_summary_full = pd.concat([df_summary, missing_df]).sort_index()
mask = df_summary_full.select_dtypes(include="number").sum(axis=1) == 0
df_summary_full.loc[mask] = np.nan


print(f"Total rows: {len(df_summary_full)}")
print(f"Rows fully zero: {mask.sum()}")
print(df_summary_full.isna().sum())

df_summary_full.to_parquet(out_folder / "ship_summary.parquet")
