import logging
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count, get_context

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from desire.utils.normalizer import CoordsNormalizer

logger = logging.getLogger(__name__)

global_nodes = None
global_sailing_trajs = None
global_mmsis_in_frame = None


def init_worker(nodes, sailing_trajs, mmsis_in_frame):
    global global_nodes, global_sailing_trajs, global_mmsis_in_frame
    global_nodes = nodes
    global_sailing_trajs = sailing_trajs
    global_mmsis_in_frame = mmsis_in_frame


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


def add_time_col(df, min_timestamp):
    # TODO make it adjust to interpolation step size
    df["time"] = ((df["timestamp"].astype(int) - min_timestamp) / 5e9).astype(int)
    return df


def load_data(path, min_date, max_date):
    df = pd.read_parquet(path)

    if max_date is not None:
        max_date = pd.to_datetime(max_date) + pd.Timedelta(days=1)
        df = df[df["timestamp"] < max_date]
    if min_date is not None:
        min_date = pd.to_datetime(min_date)
        df = df[df["timestamp"] >= min_date]

    return df.copy().reset_index(drop=True)


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
    df = df.sort_values(by="time").reset_index(drop=True)

    new_rows = []
    times = df["time"].values

    for i in range(len(times) - 1):
        current_time = times[i]
        next_time = times[i + 1]
        gap = next_time - current_time

        if gap <= 1:
            continue

        # Fill after current_time
        for j in range(1, min(steps + 1, gap)):
            row = df.iloc[i].copy()
            row["time"] = current_time + j
            new_rows.append(row)

        # Fill before next_time
        for j in range(steps, 0, -1):
            if next_time - j > current_time:
                row = df.iloc[i + 1].copy()
                row["time"] = next_time - j
                new_rows.append(row)

    # Fill after all
    current_time = times[-1]
    for j in range(1, steps + 1):
        row = df.iloc[-1].copy()
        row["time"] = current_time + j
        new_rows.append(row)

    # Fill before next_time
    current_time = times[0]
    for j in range(steps, 0, -1):
        row = df.iloc[0].copy()
        row["time"] = current_time - j
        new_rows.append(row)

    # Combine original and new
    df_extra = pd.DataFrame(new_rows)
    combined = pd.concat([df, df_extra], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"])
    combined = combined.sort_values(by="time").reset_index(drop=True)

    return combined[df.columns]


def add_rel_latlon(df: pd.DataFrame):
    # df should be sorted by time and unique mmsi
    df["group_id"] = df["time"].diff().gt(1).fillna(False).groupby(df["mmsi"]).cumsum()
    df[["rel_lat", "rel_lon"]] = df.groupby("group_id")[["lat", "lon"]].diff().fillna(0)
    return df


class LazyTrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    """
    feat_cols = {
            "sailing_vessel": 1,
            "time_diff": 60,
            "in_waterways": 1,
            "distance_shore": 500,
            "to_bow": 10,
            "to_stern": 10,
            "to_port": 10,
            "to_starboard": 10
        }
    feat_cols_to_norm = [["origin_lat", "origin_lon"], ["destination_lat", "destination_lon"]]
    """

    def __init__(
        self,
        nodes_path,
        edges_path,
        normalizer: CoordsNormalizer,
        feat_cols: dict = {},  #
        feat_cols_to_norm=[],  #
        obs_len=8,
        pred_len=12,
        max_vessels=10,
        exclude_ship_types=list(range(0, 40)),
        min_date=None,
        max_date=None,
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

        nodes = load_data(nodes_path, min_date, max_date)
        min_timestamp = nodes["timestamp"].astype(int).min()
        nodes = add_time_col(nodes, min_timestamp)
        nodes = nodes.sort_values(by=["timestamp", "traj_id"])

        exclude_mmsi = nodes[nodes["ship_type"].isin(exclude_ship_types)][
            "mmsi"
        ].unique()

        edges = load_data(edges_path, min_date, max_date)
        edges = add_time_col(edges, min_timestamp)
        mmsis_in_frame = get_in_frame_dict(edges)

        self.feature_cols = []
        for latlon in [["lat", "lon"]] + feat_cols_to_norm:
            nodes[latlon] = normalizer.normalize(nodes[latlon])
            self.feature_cols += latlon

        self.feature_cols += list(feat_cols.keys())
        nodes = nodes[["time", "mmsi", "traj_id"] + self.feature_cols]

        for key, norm_val in feat_cols.items():
            nodes[key] = nodes[key].astype(float) / norm_val

        min_cur_t = nodes["time"].min() + obs_len
        max_cur_t = nodes["time"].max() - pred_len

        for cur_t in tqdm(range(min_cur_t, max_cur_t)):
            min_t = cur_t - obs_len
            max_t = cur_t + pred_len
            nodes_t = nodes[(nodes["time"] >= min_t) & (nodes["time"] < max_t)]

            traj_len = nodes_t.groupby("mmsi").size().to_dict()
            traj_len = {mmsi: l for mmsi, l in traj_len.items() if l >= pred_len}
            valid_mmsi = list(traj_len.keys())

            full_traj_mmsi = [
                mmsi
                for mmsi, l in traj_len.items()
                if (l == obs_len + pred_len) and (mmsi not in exclude_mmsi)
            ]

            for mmsi in full_traj_mmsi:
                others = mmsis_in_frame.get((cur_t - 1, mmsi), [])
                others = [o for o in others if o in valid_mmsi][:max_vessels]
                self.items.append((cur_t, mmsi, others))

        self.data = {}
        for mmsi, group in tqdm(nodes.groupby("mmsi")):
            df = add_filled_gap_steps(group, pred_len)
            df = add_rel_latlon(df)
            df = df.set_index("time").sort_index()
            self.data[mmsi] = df

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

        # Transpose and convert to tensor
        obs_abs = torch.tensor(np.stack(obs_abs, axis=0)).permute(0, 2, 1).float()
        obs_rel = torch.tensor(np.stack(obs_rel, axis=0)).permute(0, 2, 1).float()
        pred_abs = torch.tensor(np.stack(pred_abs, axis=0)).permute(0, 2, 1).float()
        pred_rel = torch.tensor(np.stack(pred_rel, axis=0)).permute(0, 2, 1).float()

        return obs_abs, pred_abs, obs_rel, pred_rel
