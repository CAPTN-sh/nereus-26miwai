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


def extract_frames_worker(args):
    cur_t, t_range, obs_len, pred_len, max_vessels, feat_cols = args

    nodes_t = global_nodes[global_nodes["time"].isin(t_range)].copy()
    full_traj_ids = nodes_t["traj_id"].unique()

    obs_range = t_range[:obs_len]
    pred_range = t_range[obs_len:]

    frames = []
    for traj_id in full_traj_ids:
        if traj_id in global_sailing_trajs:
            continue
        cur_traj = nodes_t[nodes_t["traj_id"] == traj_id].copy()
        if len(cur_traj) < len(t_range):
            continue

        obs_traj_abs = []
        obs_traj_rel = []
        pred_traj_abs = []
        pred_traj_rel = []

        cur_mmsi = cur_traj["mmsi"].iloc[0]
        mmsis = [cur_mmsi] + list(global_mmsis_in_frame.get((cur_t - 1, cur_mmsi), []))
        for mmsi in mmsis[:max_vessels]:
            traj = nodes_t[nodes_t["mmsi"] == mmsi][["time", "lat", "lon"] + feat_cols]
            if len(traj) < len(t_range) / 2:
                continue
            traj = traj.set_index("time").reindex(t_range)
            traj = traj.ffill().bfill().reset_index()

            traj[["rel_lat", "rel_lon"]] = traj[["lat", "lon"]].diff().fillna(0)

            obs_traj = traj[traj["time"].isin(obs_range)]
            pred_traj = traj[traj["time"].isin(pred_range)]

            obs_traj_abs.append(obs_traj[["lat", "lon"]].to_numpy())
            obs_traj_rel.append(obs_traj[["rel_lat", "rel_lon"] + feat_cols].to_numpy())
            pred_traj_abs.append(pred_traj[["lat", "lon"]].to_numpy())
            pred_traj_rel.append(pred_traj[["rel_lat", "rel_lon"]].to_numpy())

        obs_traj_abs = np.stack(obs_traj_abs, axis=0)
        obs_traj_rel = np.stack(obs_traj_rel, axis=0)
        pred_traj_abs = np.stack(pred_traj_abs, axis=0)
        pred_traj_rel = np.stack(pred_traj_rel, axis=0)

        frames.append([obs_traj_abs, obs_traj_rel, pred_traj_abs, pred_traj_rel])
    return frames


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        nodes_path,
        edges_path,
        normalizer: CoordsNormalizer,
        obs_len=8,
        pred_len=12,
        max_vessels=10,
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
        super(TrajectoryDataset, self).__init__()
        self.frames = []
        self.max_vessels = max_vessels

        nodes = load_data(nodes_path, min_date, max_date)
        edges = load_data(edges_path, min_date, max_date)

        min_timestamp = nodes["timestamp"].astype(int).min()
        nodes = add_time_col(nodes, min_timestamp)
        edges = add_time_col(edges, min_timestamp)

        nodes = nodes.sort_values(by=["timestamp", "traj_id"])
        norm_cols = [
            ["lat", "lon"],
            ["origin_lat", "origin_lon"],
            ["destination_lat", "destination_lon"],
        ]
        for points in norm_cols:
            nodes[points] = normalizer.normalize(nodes[points])

        sailing_trajs = nodes[nodes["sailing_vessel"]]["traj_id"].unique()

        mmsis_in_frame = get_in_frame_dict(edges)

        """
        feat_cols_norm = {
            "sailing_vessel": 1,
            "time_diff": 60,
            "in_waterways": 1,
            "distance_shore": 500,
            "to_bow": 10,
            "to_stern": 10,
            "to_port": 10,
            "to_starboard": 10,
            "origin_lat": 1,
            "origin_lon": 1,
            "destination_lat": 1,
            "destination_lon": 1,
        }
        self.add_feats = list(feat_cols_norm.keys())
        nodes = nodes[["time", "mmsi", "traj_id", "lat", "lon"] + self.add_feats]

        for key, norm_val in feat_cols_norm.items():
            nodes[key] = nodes[key].astype(float) / norm_val
        """
        self.add_feats = []

        min_cur_t = nodes["time"].min() + obs_len
        max_cur_t = nodes["time"].max() - pred_len

        args_list = []
        for cur_t in range(min_cur_t, max_cur_t):
            t_range = list(range(cur_t - obs_len, cur_t + pred_len))
            args_list.append(
                (cur_t, t_range, obs_len, pred_len, max_vessels, self.add_feats)
            )

        with get_context("spawn").Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(nodes, sailing_trajs, mmsis_in_frame),
        ) as pool:
            all_results = list(
                tqdm(
                    pool.imap_unordered(extract_frames_worker, args_list),
                    total=len(args_list),
                )
            )

        self.frames = [frame for group in all_results for frame in group if frame]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        obs_abs, obs_rel, pred_abs, pred_rel = self.frames[index]

        # Transpose and convert to tensor
        obs_abs = torch.tensor(obs_abs).permute(0, 2, 1).float()
        obs_rel = torch.tensor(obs_rel).permute(0, 2, 1).float()
        pred_abs = torch.tensor(pred_abs).permute(0, 2, 1).float()
        pred_rel = torch.tensor(pred_rel).permute(0, 2, 1).float()

        return obs_abs, pred_abs, obs_rel, pred_rel
