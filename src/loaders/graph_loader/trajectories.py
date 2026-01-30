from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pyproj
import torch
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data, Dataset

from utils.config import SHIP_DB_PATH, STEPS_PER_MINUTE, STEP_SIZE, DATA_FOLDER_PATH

def process_time(df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    df = (
        df[df["timestamp"].between(min_date, max_date + timedelta(days=1))]
        .copy()
        .reset_index(drop=True)
    )
    df["time"] = df["timestamp"].astype("datetime64[s]").astype("int64") // STEP_SIZE
    df = df.set_index("time").sort_index()
    return df


def get_in_frame_dict(edges, max_dist=None):
    if max_dist is not None:
        edges = edges[edges["dist"] <= max_dist]
    traj_id_in_frame = (
        edges.sort_values(by=["time", "traj_id", "dist"])
        .groupby(["time", "traj_id"])["traj_id_other"]
        .apply(list)
        .to_dict()
    )
    return traj_id_in_frame


def cords_to_meters(df: pd.DataFrame):
    # TODO get CRS from config
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:25832"), always_xy=True
    )
    df["x"], df["y"] = transformer.transform(df["lon"].values, df["lat"].values)
    df = df.drop(columns = ["lon", "lat"])
    return df


class GraphTrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        flag: str,
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        feat_cols=[],
        obs_len=60,
        pred_len=30,
        min_len_in_minutes=1,
        force_rebuild=False,
        max_neighbors=5,
    ):
        super(GraphTrajectoryDataset, self).__init__()

        self.coords = pyproj.Transformer.from_crs(
            pyproj.CRS("EPSG:4326"), 
            pyproj.CRS("EPSG:25832"), 
            always_xy=True
        )

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.min_valid_window = min_len_in_minutes * STEPS_PER_MINUTE

        self.l_pad = obs_len
        self.r_pad = pred_len

        self.items = []

        self.t0 = {}
        self.pos_map = {}
        self.feat_map = {}
        self.edge_map = defaultdict(dict)

        nodes = pd.read_parquet(nodes_path)
        nodes = process_time(nodes, min_date, max_date)

        if nodes.empty:
            raise ValueError("There are no values within the given time range.")

        edges = pd.read_parquet(edges_path)
        edges = process_time(edges, min_date, max_date)
        risk_mmsi = (
            edges
            .sort_values(["collision_risk", "dist"], ascending=[False, True])
            .groupby(["time", "mmsi"], as_index=False)
            .head(max_neighbors)
        )

        traj_id = nodes.reset_index()[["time", "mmsi", "traj_id"]].copy()
        risk_mmsi = risk_mmsi.merge(traj_id, on=["time", "mmsi"], how="left")
        traj_id.columns = ["time", "mmsi_other", "traj_id_other"]
        risk_mmsi = risk_mmsi.merge(traj_id, on=["time", "mmsi_other"], how="left")

        traj_id_in_frame = get_in_frame_dict(risk_mmsi)

        nodes = nodes.fillna(0).sort_index()
        nodes = cords_to_meters(nodes)
        traj_g = nodes.groupby("traj_id")
        nodes[["rel_x", "rel_y"]] = traj_g[["x", "y"]].diff().fillna(0)

        nodes = nodes[["traj_id", "x", "y", "rel_x", "rel_y", "speed"]]
        edges = risk_mmsi[["time", "traj_id", "traj_id_other", "dist"]]

        nodes["rel_x"] = nodes["rel_x"] / 100.0
        nodes["rel_y"] = nodes["rel_y"] / 100.0
        nodes["speed"] = nodes["speed"] / 40
        edges["dist"] = edges["dist"] / 2000

        print("sample")

        for traj_id, traj in tqdm(nodes.groupby("traj_id")):
            for cur_t in traj.index[self.min_valid_window:-self.min_valid_window]:
                neighbors = traj_id_in_frame.get((cur_t, traj_id), [])
                self.items.append((cur_t, traj_id, neighbors))

        for traj_id, df in tqdm(nodes.groupby("traj_id"), desc="nodes"):
            self.t0[traj_id] = df.index[0]

            pos = df[["x", "y"]].to_numpy(dtype=np.float32)
            padded_pos = np.pad(pos, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
            self.pos_map[traj_id] = padded_pos

            feat = df[["rel_x", "rel_y", "speed"]].to_numpy(dtype=np.float32)
            padded_feat = np.pad(feat, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
            self.feat_map[traj_id] = padded_feat

        for row in tqdm(edges.itertuples(), desc="Edges"):
            self.edge_map[(row.time, row.traj_id)][row.traj_id_other] = [row.dist]


    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        cur_t, target_id, neighbors = self.items[index]

        node_ids = [target_id] + neighbors

        obs_feat_list = []
        obs_pos_list = []
        fut_feat_list = []
        fut_pos_list = []

        obs_mask_list = []
        fut_mask_list = []

        for traj_id in node_ids:
            t0 = self.t0[traj_id]
            pos = self.pos_map[traj_id]
            feat = self.feat_map[traj_id]

            t = self.l_pad + cur_t-t0

            obs_feat_list.append(feat[t-self.obs_len:t])
            obs_pos_list.append(pos[t-self.obs_len:t])
            fut_feat_list.append(feat[t:t+self.pred_len])
            fut_pos_list.append(pos[t:t+self.pred_len])

            obs_mask_list.append(np.arange(t - self.obs_len, t) >= self.l_pad)
            fut_mask_list.append(np.arange(t, t + self.pred_len) < (len(pos) - self.r_pad))

        t0 = self.t0[target_id]
        edge_index = [[],[]]
        edge_attr = []
        for a_idx, a_traj_id in enumerate(node_ids):
            edges = self.edge_map.get((cur_t, a_traj_id))
            if not edges:
                continue
            for b_idx, b_traj_id in enumerate(node_ids):
                edge = edges.get(b_traj_id)
                if not edge:
                    continue
                edge_attr.append(edge)
                edge_index[0].append(a_idx)
                edge_index[1].append(b_idx)

        is_ego = torch.zeros(len(node_ids), dtype=torch.bool)
        is_ego[0] = True

        data = Data(
            x=torch.from_numpy(np.asarray(obs_feat_list)).float(),
            x_pos=torch.from_numpy(np.asarray(obs_pos_list)).float(),
            obs_mask=torch.from_numpy(np.asarray(obs_mask_list)).bool(),
            edge_index=torch.from_numpy(np.asarray(edge_index)).long(),
            edge_attr=torch.from_numpy(np.asarray(edge_attr)).float(),
            y=torch.from_numpy(np.asarray(fut_feat_list)).float(),
            y_pos=torch.from_numpy(np.asarray(fut_pos_list)).float(),
            fut_mask=torch.from_numpy(np.asarray(fut_mask_list)).bool(),
            is_ego = is_ego,
        )

        return data
    
    def _pad(self, x):
        return np.pad(x, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')


if __name__ == "__main__":
    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"

    flag="train"
    file_name = f"fh_kiel_{flag}"
    dset = GraphTrajectoryDataset(
        nodes_path=data_folder / f"{file_name}_ship_features.parquet",
        edges_path=data_folder / f"{file_name}_ship2ship_features.parquet",
        flag=flag,
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        feat_cols=["speed"],
        pred_len=30,
        obs_len=60,
    )

    print("Num samples:", len(dset))