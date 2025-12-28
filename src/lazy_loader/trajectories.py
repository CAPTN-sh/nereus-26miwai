import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def seq_collate(data):
    (obs_feat_seq, obs_pos_seq, obs_pos_rel_seq, fut_pos_seq, fut_pos_rel_seq) = zip(
        *data
    )

    _len = [len(seq) for seq in obs_pos_seq]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    # [B, C, T] (batch, channels, time)
    obs_feat = torch.cat(obs_feat_seq, dim=0)
    obs_pos = torch.cat(obs_pos_seq, dim=0)
    obs_pos_rel = torch.cat(obs_pos_rel_seq, dim=0)
    fut_pos = torch.cat(fut_pos_seq, dim=0)
    fut_pos_rel = torch.cat(fut_pos_rel_seq, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    return obs_feat, obs_pos, obs_pos_rel, fut_pos, fut_pos_rel, seq_start_end


def get_interp_step_size(df: pd.DataFrame):
    first_traj = df.loc[df["traj_id"] == df.iloc[0]["traj_id"]].copy()
    ts = first_traj.sort_values("timestamp")["timestamp"]
    step_size = int((ts.iloc[1] - ts.iloc[0]).total_seconds())
    return step_size


def process_time(
    df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp, step_size: int
) -> pd.DataFrame:
    df = (
        df[df["timestamp"].between(min_date, max_date + timedelta(days=1))]
        .copy()
        .reset_index(drop=True)
    )
    df["time"] = df["timestamp"].astype("datetime64[s]").astype("int64") // step_size
    df = df.set_index("time").sort_index()
    return df


def get_in_frame_dict(edges, max_dist=None):
    if max_dist is not None:
        edges = edges[edges["dist"] <= max_dist]
    mmsis_in_frame = (
        edges.sort_values(by=["time", "mmsi", "dist"])
        .groupby(["time", "mmsi"])["mmsi_other"]
        .apply(list)
        .to_dict()
    )
    return mmsis_in_frame


def add_filled_gap_steps(df: pd.DataFrame, steps: int):
    df = df.sort_index()

    new_rows = []
    times = df.index

    for i in range(len(times) - 1):
        current_time = times[i]
        next_time = times[i + 1]
        gap = next_time - current_time

        if gap <= 1:
            continue

        # Fill after current_time
        for j in range(1, min(steps + 1, gap)):
            row = df.iloc[i].copy()
            row.name = current_time + j
            new_rows.append(row)

        # Fill before next_time
        for j in range(steps, 0, -1):
            if next_time - j > current_time:
                row = df.iloc[i + 1].copy()
                row.name = next_time - j
                new_rows.append(row)

    # Fill after all
    current_time = times[-1]
    for j in range(1, steps + 1):
        row = df.iloc[-1].copy()
        row.name = current_time + j
        new_rows.append(row)

    # Fill before next_time
    current_time = times[0]
    for j in range(steps, 0, -1):
        row = df.iloc[0].copy()
        row.name = current_time - j
        new_rows.append(row)

    # Combine original and new
    combined = pd.concat([df, pd.DataFrame(new_rows)])
    combined = combined[~combined.index.duplicated()].sort_index()

    return combined


def add_rel_pos(df: pd.DataFrame):
    # df hast to be sorted by time and unique mmsi prior to function call!
    df["group_id"] = df.index.to_series().diff().gt(1).cumsum()
    df[["rel_x", "rel_y"]] = df.groupby("group_id")[["x", "y"]].diff().fillna(0)
    return df


def cords_to_meters(df: pd.DataFrame):
    # TODO get CRS from config
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:25832"), always_xy=True
    )
    df["x"], df["y"] = transformer.transform(df["lon"].values, df["lat"].values)
    return df


class LazyTrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        feat_cols=[],
        obs_len=8,
        pred_len=12,
        max_neighbors=10,
        exclude_ship_types=list(range(0, 40)),
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(LazyTrajectoryDataset, self).__init__()
        self.items = []
        self.data = {}
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.feature_cols = feat_cols

        nodes = pd.read_parquet(nodes_path)

        step_size = get_interp_step_size(nodes)
        nodes = process_time(nodes, min_date, max_date, step_size)

        if nodes.empty:
            raise ValueError("There are no values within the given time range.")

        # TODO config
        ship_db_path = Path("data/ais/ship_db/ship_db.parquet")
        ship_db = pd.read_parquet(ship_db_path)
        exclude_mmsi = ship_db[ship_db["ship_type"].isin(exclude_ship_types)]["mmsi"]

        nodes = nodes[["mmsi", "lat", "lon"] + self.feature_cols]
        nodes = cords_to_meters(nodes)
        self.feature_cols = [
            col.replace("lon", "x").replace("lat", "y") for col in self.feature_cols
        ]

        edges = pd.read_parquet(edges_path)
        edges = process_time(edges, min_date, max_date, step_size)
        mmsis_in_frame = get_in_frame_dict(edges)

        for cur_t in tqdm(range(nodes.index[0], nodes.index[-1])):
            self._add_items_at_t(
                nodes, cur_t, exclude_mmsi, mmsis_in_frame, max_neighbors
            )

        for mmsi, group in tqdm(nodes.groupby("mmsi")):
            df = add_filled_gap_steps(group, pred_len)
            df = add_rel_pos(df)
            self.data[mmsi] = df

    def _add_items_at_t(
        self, nodes, cur_t, exclude_mmsi, mmsis_in_frame, max_neighbors
    ):
        nodes_t = nodes.loc[cur_t - self.obs_len : cur_t + self.pred_len - 1]
        if nodes_t.empty:
            return

        traj_len = nodes_t.groupby("mmsi").size().to_dict()
        traj_len = {mmsi: l for mmsi, l in traj_len.items() if l >= self.pred_len}
        valid_mmsi = list(traj_len.keys())

        full_traj_mmsi = [
            mmsi
            for mmsi, len in traj_len.items()
            if (len == self.obs_len + self.pred_len) and (mmsi not in exclude_mmsi)
        ]

        for mmsi in full_traj_mmsi:
            others = mmsis_in_frame.get((cur_t - 1, mmsi), [])
            others = [o for o in others if o in valid_mmsi]
            if max_neighbors is not None:
                others = others[:max_neighbors]
            self.items.append((cur_t, mmsi, others))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        cur_t, cur_mmsi, others = self.items[index]

        obs_feat = []
        obs_pos = []
        obs_pos_rel = []
        fut_pos = []
        fut_pos_rel = []

        for mmsi in [cur_mmsi] + others:
            df = self.data[mmsi]
            obs_df = df.loc[cur_t - self.obs_len : cur_t - 1]
            fut_df = df.loc[cur_t : cur_t + self.pred_len - 1]

            obs_feat.append(obs_df[self.feature_cols].values)
            obs_pos.append(obs_df[["x", "y"]].values)
            obs_pos_rel.append(obs_df[["rel_x", "rel_y"]].values)
            fut_pos.append(fut_df[["x", "y"]].values)
            fut_pos_rel.append(fut_df[["rel_x", "rel_y"]].values)

        obs_feat = self._to_tensor(obs_feat)
        obs_pos = self._to_tensor(obs_pos)
        obs_pos_rel = self._to_tensor(obs_pos_rel)
        fut_pos = self._to_tensor(fut_pos)
        fut_pos_rel = self._to_tensor(fut_pos_rel)

        return obs_feat, obs_pos, obs_pos_rel, fut_pos, fut_pos_rel

    def _to_tensor(self, traj):
        return torch.tensor(np.stack(traj, axis=0)).permute(0, 2, 1).float()
