from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pyproj
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from loaders.graph_loader.normalizer import normalize

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

        self.transformer = pyproj.Transformer.from_crs(
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
        self.data = {}
        self.traj_id_to_idx = {}

        self.t0 = []
        self.node_feat = []
        self.node_feat_st = []
        self.edge_feat = {}
        self.edge_feat_st = {}

        print("loading")

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

        print("merging")

        traj_id = nodes.reset_index()[["time", "mmsi", "traj_id"]].copy()
        risk_mmsi = risk_mmsi.merge(traj_id, on=["time", "mmsi"], how="left")
        traj_id.columns = ["time", "mmsi_other", "traj_id_other"]
        risk_mmsi = risk_mmsi.merge(traj_id, on=["time", "mmsi_other"], how="left")

        traj_id_in_frame = get_in_frame_dict(risk_mmsi)

        print("norm")

        nodes = nodes.fillna(0).sort_index()
        nodes["x"], nodes["y"] = self.transformer.transform(
            nodes["lon"].values, 
            nodes["lat"].values
        )
        nodes = nodes[["traj_id", "x", "y", "speed", "course"]]

        nodes["x_st"] = (nodes["x"] - 573663) / (586057 - 573663)
        nodes["y_st"] = (nodes["y"] - 6018805) / (6035410 - 6018805)
        nodes["speed_st"] = nodes["speed"] / 40
        nodes["course_st"] = nodes["course"] / 360

        edges = risk_mmsi[["time", "traj_id", "traj_id_other", "dist"]].set_index("time").sort_index()
        edges["dist_st"] = edges["dist"] / 2000

        print("sample")

        for traj_id, traj in tqdm(nodes.groupby("traj_id")):
            for cur_t in traj.index[self.min_valid_window:-self.min_valid_window]:
                neighbors = traj_id_in_frame.get((cur_t, traj_id), [])
                self.items.append((cur_t, traj_id, neighbors))

        for idx, (traj_id, df) in enumerate(tqdm(nodes.groupby("traj_id"), desc="nodes")):
            self.traj_id_to_idx[traj_id] = idx
            self.t0.append(df.index[0])
            self.node_feat.append(self._pad(df[["x", "y"] + feat_cols].to_numpy(dtype=np.float32)))
            self.node_feat_st.append(self._pad(df[["x_st", "y_st"] + [f"{c}_st" for c in feat_cols]].to_numpy(dtype=np.float32)))

        for (traj_id, traj_id_other), df in tqdm(edges.groupby(["traj_id", "traj_id_other"]), desc="Edges"):
            t0 = self.t0[self.traj_id_to_idx[traj_id]]
            tN = t0 + len(self.node_feat[self.traj_id_to_idx[traj_id]]) - self.l_pad - self.r_pad

            full_index = np.arange(t0, tN)
            df = df.reindex(full_index, fill_value=0)

            dist = df["dist"].to_numpy(dtype=np.float32)[:, None]
            dist_st = df["dist_st"].to_numpy(dtype=np.float32)[:, None]

            self.edge_feat[(traj_id, traj_id_other)] = self._pad(dist)
            self.edge_feat_st[(traj_id, traj_id_other)] = self._pad(dist_st)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        cur_t, traj_id, neighbors = self.items[index]

        idx = self.traj_id_to_idx[traj_id]
        t_0 = self.t0[idx]
        node_feat = self.node_feat[idx]
        node_feat_st = self.node_feat_st[idx]

        hist_len = cur_t - t_0
        first_history_index = torch.tensor([max(0, self.obs_len - hist_len)])

        cur_idx = self.l_pad + hist_len

        x_t = node_feat[cur_idx - self.obs_len : cur_idx]
        y_t = node_feat[cur_idx : cur_idx + self.pred_len, :2]
        x_st_t = node_feat_st[cur_idx - self.obs_len : cur_idx]
        y_st_t = node_feat_st[cur_idx : cur_idx + self.pred_len, :2]

        neighbors_data_st = []
        neighbors_edge_value = []
        for nn_traj_id in neighbors:
            nn_idx = self.traj_id_to_idx[nn_traj_id]
            
            # Neighbor node state
            nn_node_feat_st = self.node_feat_st[nn_idx]
            nn_traj = nn_node_feat_st[cur_idx - self.obs_len : cur_idx]
            neighbors_data_st.append(torch.from_numpy(nn_traj).unsqueeze(0))

            # Edge feature (traj_id -> nn_traj_id)
            edge_feat = self.edge_feat_st[(traj_id, nn_traj_id)]
            edge_slice = edge_feat[cur_idx - self.obs_len : cur_idx]
            neighbors_edge_value.append(torch.from_numpy(edge_slice).unsqueeze(0))

        x_t = self._to_tensor(x_t)
        y_t = self._to_tensor(y_t)
        x_st_t = self._to_tensor(x_st_t)
        y_st_t = self._to_tensor(y_st_t)

        return first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st, neighbors_edge_value, None, None

    def _to_tensor(self, traj, dtype=torch.float32):
        return torch.from_numpy(traj).unsqueeze(0).to(dtype)
    
    def _pad(self, x):
        return np.pad(x, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')


if __name__ == "__main__":
    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"

    file_name = "fh_kiel_val"
    dset = GraphTrajectoryDataset(
        nodes_path=data_folder / f"{file_name}_ship_features.parquet",
        edges_path=data_folder / f"{file_name}_ship2ship_features.parquet",
        flag="val",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        feat_cols=["speed", "course"],
        pred_len=30,
        obs_len=60,
    )

    print("Num samples:", len(dset))
    print("Num trajectories:", len(dset.traj_id_to_idx))