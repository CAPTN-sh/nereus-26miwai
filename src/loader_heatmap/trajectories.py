from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.config import SHIP_DB_PATH, STEPS_PER_MINUTE, STEP_SIZE

def process_time(df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    df = (
        df[df["timestamp"].between(min_date, max_date + timedelta(days=1))]
        .copy()
        .reset_index(drop=True)
    )
    df["time"] = df["timestamp"].astype("datetime64[s]").astype("int64") // STEP_SIZE
    df = df.set_index("time").sort_index()
    return df

def cords_to_meters(df: pd.DataFrame):
    # TODO get CRS from config
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:25832"), always_xy=True
    )
    df["x"], df["y"] = transformer.transform(df["lon"].values, df["lat"].values)
    return df

class TrajectoryHeatmapDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        nodes_path: Path,
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        feat_cols=[],
        obs_len=60,
        fut_len=120,
        min_len_in_minutes=1,
        normalize = False,
    ):
        super(TrajectoryHeatmapDataset, self).__init__()

        min_valid_window = min_len_in_minutes * STEPS_PER_MINUTE

        feat_cols = feat_cols.copy()
        self.items = []
        self.feat_map = {}
        self.pos_map = {}
        self.rel_map = {}
        self.obs_len = obs_len
        self.fut_len = fut_len
        self.l_pad = max(0, obs_len-min_valid_window)
        self.r_pad = max(0, fut_len-min_valid_window)

        nodes = pd.read_parquet(nodes_path)

        # remove traj without destination
        nodes = process_time(nodes, min_date, max_date)

        nodes = nodes[nodes.groupby('traj_id')['is_entry'].transform('last') == 1]

        if nodes.empty:
            raise ValueError("There are no values within the given time range.")

        ship_db = pd.read_parquet(SHIP_DB_PATH)
        nodes = nodes.reset_index().merge(ship_db, on="mmsi")
        nodes = cords_to_meters(nodes)

        nodes["length"] = np.log1p(nodes['to_bow'] + nodes['to_stern']) / np.log1p(400)
        nodes["width"] = np.log1p(nodes['to_port'] + nodes['to_starboard']) / np.log1p(60)
        nodes["sailing"] = nodes["ship_group"] == "sailing"
        nodes["cargo"] = nodes["ship_group"] == "cargo"
        nodes["passenger"] = nodes["ship_group"] == "passenger"

        feat_cols += ["acc", "angular_difference", "length",  "width",  "sailing", "cargo", "passenger"]

        if normalize:
            nodes["speed"] = nodes["speed"] / 40
            nodes["acc"] = nodes["acc"] / 4

            for col in ['course', 'angular_difference']:
                rad = np.deg2rad(nodes[col])
                nodes[col + "_sin"] = np.sin(rad)
                nodes[col + "_cos"] = np.cos(rad)
                feat_cols += [col + "_sin", col + "_cos"]
                feat_cols.remove(col)


        for traj_id, group in tqdm(nodes.groupby("traj_id")):
            group = group.sort_values("time").reset_index(drop=True)

            feat = group[feat_cols].to_numpy(dtype=np.float32)
            padded_feat = np.pad(feat, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
            self.feat_map[traj_id] = padded_feat

            pos = group[["x", "y"]].to_numpy(dtype=np.float32)
            padded_pos = np.pad(pos, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
            self.pos_map[traj_id] = padded_pos

            rel = np.diff(pos, axis=0, prepend=pos[0:1]) 
            padded_rel = np.pad(rel, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
            self.rel_map[traj_id] = padded_rel

            for cur_t in range(obs_len, len(padded_pos) - fut_len):
                self.items.append((cur_t, traj_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        cur_t, traj_id = self.items[index]

        feat = self.feat_map[traj_id]
        abs_pos = self.pos_map[traj_id]

        obs_feat = torch.from_numpy(feat[cur_t - self.obs_len : cur_t]).float()
        obs_pos = torch.from_numpy(abs_pos[cur_t - self.obs_len : cur_t]).float()
        fut_pos = torch.from_numpy(abs_pos[cur_t : cur_t + self.fut_len]).float()
        fin_pos = torch.from_numpy(abs_pos[-(self.r_pad + 1)]).float()

        obs_mask = torch.arange(cur_t - self.obs_len, cur_t) >= self.l_pad
        fut_mask = torch.arange(cur_t, cur_t + self.fut_len) < (len(abs_pos) - self.r_pad)
        
        rel_pos = self.rel_map[traj_id]
        obs_rel = torch.from_numpy(rel_pos[cur_t - self.obs_len : cur_t]).float()
        fut_rel = torch.from_numpy(rel_pos[cur_t : cur_t + self.fut_len]).float()
        
        if self.l_pad >= cur_t - self.obs_len:
            bad_value = self.l_pad - cur_t + self.obs_len
            obs_mask[bad_value] = False

        return obs_feat, obs_pos, obs_rel, obs_mask, fut_pos, fut_rel, fut_mask, fin_pos