from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pyproj
import torch
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import joblib
import time as timecounter

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

        cache_key = f"sl_{min_date.date()}_{max_date.date()}_{obs_len}_{pred_len}_{flag}"
        cache_path = Path("data/cache") / f"graph_loader_{cache_key}.pt"

        if cache_path.exists() and not force_rebuild:
            print("loading from cache")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.__dict__.update(cache)
            return

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

        self.num_node_feats = 0
        self.num_edge_feats = 0

        nodes = pd.read_parquet(nodes_path)
        nodes = process_time(nodes, min_date, max_date)

        if nodes.empty:
            raise ValueError("There are no values within the given time range.")
        
        ship_db = pd.read_parquet(SHIP_DB_PATH)
        nodes = nodes.reset_index().merge(ship_db, on="mmsi")

        edges_all = pd.read_parquet(edges_path)
        edges_all = process_time(edges_all, min_date, max_date)
        edges = (
            edges_all
            .sort_values(["collision_risk", "dist"], ascending=[False, True])
            .groupby(["time", "mmsi"], as_index=False)
            .head(max_neighbors)
        )

        traj_info = nodes[["time", "mmsi", "traj_id"]].copy().sort_values(["time"])
        edges = edges.merge(traj_info, on=["time", "mmsi"], how="inner")

        traj_info.columns = ["time", "mmsi_other", "traj_id_other"]
        traj_size = traj_info.groupby("traj_id_other")["time"].transform("size")
        traj_info["n_obs"] = (traj_info.groupby("traj_id_other").cumcount() + 1)
        traj_info["n_fut"] = traj_size - traj_info["n_obs"]

        edges = edges.merge(traj_info, on=["time", "mmsi_other"], how="inner")

        edges = edges[
            (edges["n_obs"] >= self.min_valid_window) &
            (edges["n_fut"] >= self.min_valid_window)
        ]

        traj_id_in_frame = get_in_frame_dict(edges)

        nodes = nodes.set_index("time").fillna(0).sort_index()
        nodes = cords_to_meters(nodes)
        traj_g = nodes.groupby("traj_id")
        nodes[["rel_x", "rel_y"]] = traj_g[["x", "y"]].diff().fillna(0)

        def norm_deg(df, col, cols):
            rad = np.deg2rad(df[col])
            df[col + "_sin"] = np.sin(rad)
            df[col + "_cos"] = np.cos(rad)
            cols.append(col + "_sin")
            cols.append(col + "_cos")
            cols.remove(col)
            return df, cols
        
        node_feat_cols = ["rel_x", "rel_y", "speed", "course", "acc", "angular_difference", "length",  "width",  "sailing", "cargo", "passenger", "hour_of_day"]

        nodes["rel_x"] = nodes["rel_x"] / 100.0
        nodes["rel_y"] = nodes["rel_y"] / 100.0
        nodes["speed"] = nodes["speed"] / 40.0
        nodes["acc"] = nodes["acc"] / 4.0
        nodes["length"] = np.log1p(nodes['to_bow'] + nodes['to_stern']) / np.log1p(400)
        nodes["width"] = np.log1p(nodes['to_port'] + nodes['to_starboard']) / np.log1p(60)
        nodes["sailing"] = nodes["ship_group"] == "sailing"
        nodes["cargo"] = nodes["ship_group"] == "cargo"
        nodes["passenger"] = nodes["ship_group"] == "passenger"
        nodes["hour_of_day"] = nodes["timestamp"].dt.hour / 24 * 360

        nodes, node_feat_cols = norm_deg(nodes, 'hour_of_day', node_feat_cols)
        nodes, node_feat_cols = norm_deg(nodes, 'course', node_feat_cols)
        nodes, node_feat_cols = norm_deg(nodes, 'angular_difference', node_feat_cols)

        self.num_node_feats = len(node_feat_cols)
        nodes = nodes[["traj_id", "x", "y"] + node_feat_cols].copy()

        edge_feat_cols = ["dist", 'rel_speed', 'course_diff', 'rel_bearing', 'tcpa', 'dcpa', 'collision_risk']

        edges["dist"] = edges["dist"] / 2000
        edges["rel_speed"] = edges["rel_speed"] / 40
        edges["tcpa"] = edges["tcpa"] / 3600
        edges["dcpa"] = edges["dcpa"] / 2000
        edges, edge_feat_cols = norm_deg(edges, 'course_diff', edge_feat_cols)
        edges, edge_feat_cols = norm_deg(edges, 'rel_bearing', edge_feat_cols)
        
        self.num_edge_feats = len(edge_feat_cols)
        edges = edges.fillna(0)[["time", "traj_id", "traj_id_other"] + edge_feat_cols].copy()

        for traj_id, traj in tqdm(nodes.groupby("traj_id"), desc="Nodes"):
            for cur_t in traj.index[self.min_valid_window:-self.min_valid_window]:
                neighbors = traj_id_in_frame.get((cur_t, traj_id), [])
                self.items.append((cur_t, traj_id, neighbors))

            self.t0[traj_id] = traj.index[0]

            pos = traj[["x", "y"]].to_numpy(dtype=np.float32)
            padded_pos = np.pad(pos, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
            self.pos_map[traj_id] = padded_pos

            feat = traj[node_feat_cols].to_numpy(dtype=np.float32)
            padded_feat = np.pad(feat, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
            self.feat_map[traj_id] = padded_feat

        for row in tqdm(edges.itertuples(index=False, name=None), total=len(edges), desc="Edges"):
            time, traj_id, traj_id_other = row[:3]
            self.edge_map[(time, traj_id)][traj_id_other] = row[3:]


        torch.save(
            {
                "items":self.items,
                "t0":self.t0,
                "pos_map":self.pos_map,
                "feat_map":self.feat_map,
                "edge_map":self.edge_map,
                "obs_len":self.obs_len,
                "pred_len":self.pred_len,
                "l_pad":self.l_pad,
                "r_pad":self.r_pad,
                "num_node_feats":self.num_node_feats,
                "num_edge_feats":self.num_edge_feats,
            },
            cache_path,
        )


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

        if len(edge_attr) == 0:
            edge_attr = torch.empty((0, self.num_edge_feats), dtype=torch.float)
        else:
            edge_attr = torch.from_numpy(np.asarray(edge_attr)).float()


        lengths = np.asarray(obs_mask_list).sum(axis=1)

        if np.any(lengths == 0):
            print("ZERO-LENGTH OBS FOUND")
            print("cur_t:", cur_t)
            print("target_id:", target_id)
            print("node_ids:", node_ids)
            print("lengths:", lengths)
            print("t indices:", [
                (traj_id, self.t0[traj_id], cur_t)
                for traj_id in node_ids
            ])
            print("last_obs_pos:",
                [(traj_id,
                    obs_pos_list[i][obs_mask_list[i]].tolist()[-1]
                    if obs_mask_list[i].any() else None)
                for i, traj_id in enumerate(node_ids)])

            raise RuntimeError("Found node with zero valid observations")

        data = Data(
            x=torch.from_numpy(np.asarray(obs_feat_list)).float(),
            x_pos=torch.from_numpy(np.asarray(obs_pos_list)).float(),
            obs_mask=torch.from_numpy(np.asarray(obs_mask_list)).bool(),
            edge_index=torch.from_numpy(np.asarray(edge_index)).long(),
            edge_attr=edge_attr,
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
    file_name = f"fh_{data_folder.name}_val"
    dset = GraphTrajectoryDataset(
        nodes_path=data_folder / f"{file_name}_ship_features.parquet",
        edges_path=data_folder / f"{file_name}_ship2ship_features.parquet",
        flag="val",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2023-01-01"),
        feat_cols=[],
        pred_len=30,
        obs_len=60,
        force_rebuild=False,
    )

    loader = DataLoader(
        dset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        pin_memory=False,
        prefetch_factor=4,
        persistent_workers=False,
        drop_last=True,
    )

    for batch in tqdm(loader):
        a = 0