import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from desire.utils.normalizer import Normalizer

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    return obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, seq_start_end


def get_interp_step_size(df: pd.DataFrame):
    first_traj = df.loc[df["traj_id"] == df.iloc[0]["traj_id"]].copy()
    ts = first_traj.sort_values("timestamp")["timestamp"]
    step_size = int((ts.iloc[1] - ts.iloc[0]).total_seconds())
    return step_size


def process_time(
    df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp, step_size: int
) -> pd.DataFrame:
    df = df[df["timestamp"].between(min_date, max_date)].copy().reset_index(drop=True)
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


def add_rel_latlon(df: pd.DataFrame):
    # df hast to be sorted by time and unique mmsi prior to function call!
    df["group_id"] = df.index.to_series().diff().gt(1).cumsum()
    df[["rel_lat", "rel_lon"]] = df.groupby("group_id")[["lat", "lon"]].diff().fillna(0)
    return df


class LazyTrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    """
    feat_cols = {
            "sailing_vessel": 1,
            "time_diff": 60,
            "in_waterways": 1,
            "distance_shore": 500, # max distance in kiel shore to shore
            "to_bow": 10, # max distnace
            "to_stern": 10,
            "to_port": 10,
            "to_starboard": 10
        }

        # use normalizer with mean and std and also save (max min)

    feat_cols_to_norm = [["origin_lat", "origin_lon"], ["destination_lat", "destination_lon"]]
    """

    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        normalizer_path: Path,
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        feat_cols=[],
        obs_len=8,
        pred_len=12,
        max_vessels=10,
        exclude_ship_types=list(range(0, 40)),
        num_workers=8,
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
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.normalizer = Normalizer()

        nodes = pd.read_parquet(nodes_path)

        norm_cols = ["lat", "lon"] + feat_cols
        if normalizer_path.exists():
            self.normalizer.load_from_file(normalizer_path)
        else:
            self.normalizer.approximate_from_df(nodes, norm_cols)
            self.normalizer.save_to_file(normalizer_path)

        step_size = get_interp_step_size(nodes)
        nodes = process_time(nodes, min_date, max_date, step_size)

        if nodes.empty:
            raise ValueError("There are no values within the given time range.")

        for col in norm_cols:
            norm_by = "lat" if "lat" in col else "lon" if "lon" in col else col
            nodes[col] = self.normalizer.normalize(nodes[col].astype(float), norm_by)

        exclude_mmsi = nodes[nodes["ship_type"].isin(exclude_ship_types)][
            "mmsi"
        ].unique()

        self.feature_cols = feat_cols
        nodes = nodes[["mmsi", "lat", "lon"] + self.feature_cols]

        edges = pd.read_parquet(edges_path)
        edges = process_time(edges, min_date, max_date, step_size)
        mmsis_in_frame = get_in_frame_dict(edges)

        for cur_t in tqdm(range(nodes.index[0], nodes.index[-1])):
            self._add_items_at_t(
                nodes, cur_t, exclude_mmsi, mmsis_in_frame, max_vessels
            )

        self.data = {}
        for mmsi, group in tqdm(nodes.groupby("mmsi")):
            df = add_filled_gap_steps(group, pred_len)
            df = add_rel_latlon(df)
            self.data[mmsi] = df

    def _add_items_at_t(self, nodes, cur_t, exclude_mmsi, mmsis_in_frame, max_vessels):
        nodes_t = nodes.loc[cur_t - self.obs_len : cur_t + self.pred_len - 1]
        if nodes_t.empty:
            return

        traj_len = nodes_t.groupby("mmsi").size().to_dict()
        traj_len = {mmsi: l for mmsi, l in traj_len.items() if l >= self.pred_len}
        valid_mmsi = list(traj_len.keys())

        full_traj_mmsi = [
            mmsi
            for mmsi, l in traj_len.items()
            if (l == self.obs_len + self.pred_len) and (mmsi not in exclude_mmsi)
        ]

        for mmsi in full_traj_mmsi:
            others = mmsis_in_frame.get((cur_t - 1, mmsi), [])
            others = [o for o in others if o in valid_mmsi]
            if max_vessels is not None:
                others = others[:max_vessels]
            self.items.append((cur_t, mmsi, others))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        cur_t, cur_mmsi, others = self.items[index]

        obs_abs = []
        obs_rel = []
        pred_abs = []
        pred_rel = []

        for mmsi in [cur_mmsi] + others:
            df = self.data[mmsi]
            obs_traj = df.loc[cur_t - self.obs_len : cur_t - 1]
            pred_traj = df.loc[cur_t : cur_t + self.pred_len - 1]

            obs_abs.append(obs_traj[["lat", "lon"]].values)
            obs_rel.append(obs_traj[["rel_lat", "rel_lon"] + self.feature_cols].values)
            pred_abs.append(pred_traj[["lat", "lon"]].values)
            pred_rel.append(pred_traj[["rel_lat", "rel_lon"]].values)

        obs_abs = self._to_tensor(obs_abs)
        obs_rel = self._to_tensor(obs_rel)
        pred_abs = self._to_tensor(pred_abs)
        pred_rel = self._to_tensor(pred_rel)

        return obs_abs, pred_abs, obs_rel, pred_rel

    def _to_tensor(self, traj):
        return torch.tensor(np.stack(traj, axis=0)).permute(0, 2, 1).float()
