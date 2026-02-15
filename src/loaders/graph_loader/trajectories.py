from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pyproj
import torch
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data, Dataset
from models.traisformer.params import TraisformerParams
from models.utils.maps.rasterize import Rasterizer

from utils.config import SHIP_DB_PATH, STEPS_PER_MINUTE, STEP_SIZE, DATA_FOLDER_PATH
RASTER = Rasterizer(TraisformerParams().bbox)

def process_time(df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    df = (
        df[df["timestamp"].between(min_date, max_date + timedelta(days=1))]
        .copy()
        .reset_index(drop=True)
    )
    df["time"] = df["timestamp"].astype("datetime64[s]").astype("int64") // STEP_SIZE
    return df.set_index("time").sort_index()


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
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        obs_len=60,
        pred_len=30,
        min_len_in_minutes=1,
        max_edge_dist=500,
    ):
        super(GraphTrajectoryDataset, self).__init__()

        self.coords = pyproj.Transformer.from_crs(
            pyproj.CRS("EPSG:4326"), 
            pyproj.CRS("EPSG:25832"), 
            always_xy=True
        )

        self.obs_len = self.l_pad = obs_len
        self.pred_len = self.r_pad = pred_len
        self.min_valid_window = min_len_in_minutes * STEPS_PER_MINUTE

        node_feat_cols = ["speed", "course", "acc", "angular_difference"]
        edge_feat_cols = ["dist", "rel_speed", 'course_diff', 'rel_bearing', "tcpa", "dcpa", 'collision_risk']
        static_feat_cols = ["ship_group", 'to_bow', 'to_stern', 'to_port', 'to_starboard']
        self.num_node_feats = 0
        self.num_edge_feats = 0
        self.num_static_feats = 0

        self.items = []

        self.t0 = {}
        self.pos_map = {}
        self.feat_map = {}
        self.raw_map = {}
        self.static_map = {}
        self.edge_map = defaultdict(dict)
        self.fin_pos_mask = {}

        def norm_deg(df, col, cols):
            rad = np.deg2rad(df[col])
            df[col + "_sin"] = np.sin(rad)
            df[col + "_cos"] = np.cos(rad)
            cols.append(col + "_sin")
            cols.append(col + "_cos")
            cols.remove(col)
            return df, cols

        #### NODES ####
        nodes = pd.read_parquet(nodes_path)
        # nodes = nodes[nodes.groupby('traj_id')['is_entry'].transform('last') == 1]
        nodes = process_time(nodes, min_date, max_date)
        nodes = cords_to_meters(nodes)
        traj_g = nodes.groupby("traj_id")
        nodes[["rel_x", "rel_y"]] = traj_g[["x", "y"]].diff()
        nodes = nodes.sort_index().fillna(0)

        for traj_id, traj in tqdm(nodes.groupby("traj_id"), desc="Nodes"):
            self.t0[traj_id] = traj.index[0]

            pos = traj[["x", "y"]].to_numpy(dtype=np.float32)
            padded_pos = np.pad(pos, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
            self.pos_map[traj_id] = padded_pos

            raw = traj[node_feat_cols].to_numpy(dtype=np.float32)
            padded_raw = np.pad(raw, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
            self.raw_map[traj_id] = padded_raw

            self.fin_pos_mask[traj_id] = np.array(traj['is_entry'].iloc[-1], dtype=bool)

        nodes["rel_x"] /= 100.0
        nodes["rel_y"] /= 100.0
        nodes["speed"] /= 40.0
        nodes["acc"] /= 4.0

        nodes, node_feat_cols = norm_deg(nodes, 'course', node_feat_cols)
        nodes, node_feat_cols = norm_deg(nodes, 'angular_difference', node_feat_cols)

        self.num_node_feats = 2 + len(node_feat_cols)

        #### STATIC ####
        ship_db = pd.read_parquet(SHIP_DB_PATH)
        traj_id_map = nodes[["mmsi", "traj_id"]].drop_duplicates()
        ship_db = traj_id_map.merge(ship_db, on="mmsi", how="left")

        ship_db["sailing"] = ship_db["ship_group"] == "sailing"
        ship_db["cargo"] = ship_db["ship_group"] == "cargo"
        ship_db["passenger"] = ship_db["ship_group"] == "passenger"
        ship_db["other"] = ship_db["ship_group"] == "other"
        ship_db['to_bow'] /= 100
        ship_db['to_stern'] /= 100
        ship_db['to_port'] /= 60
        ship_db['to_starboard'] /= 60

        if "ship_group" in static_feat_cols:
            static_feat_cols += ["sailing", "cargo", "passenger", "other"]
            static_feat_cols.remove("ship_group")

        ship_db = ship_db[["traj_id"] + static_feat_cols].copy()
        self.num_static_feats = len(static_feat_cols)

        self.static_map = {
            traj_id: row.to_numpy(dtype=np.float32)
            for traj_id, row in ship_db.set_index("traj_id")[static_feat_cols].iterrows()
        }

        #### EDGES ####
        if max_edge_dist == 0:
            for traj_id, traj in tqdm(nodes.groupby("traj_id"), desc="Nodes"):
                for cur_t in traj.index[self.min_valid_window:-self.min_valid_window]:
                    self.items.append((cur_t, traj_id, []))

                feat = traj[["rel_x", "rel_y"] + node_feat_cols].to_numpy(dtype=np.float32)
                padded_feat = np.pad(feat, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
                self.feat_map[traj_id] = padded_feat
            return
        
        edges_all = pd.read_parquet(edges_path)
        edges_all = process_time(edges_all, min_date, max_date)
        edges = edges_all[edges_all["dist"] <= max_edge_dist]

        traj_info = nodes.reset_index()[["time", "mmsi", "traj_id"]].copy()
        ship_info = ship_db.copy()
        edges = edges.merge(traj_info, on=["time", "mmsi"], how="inner")
        edges = edges.merge(ship_info, on=["traj_id"], how="inner")
        
        traj_info.columns = ["time", "mmsi_other", "traj_id_other"]
        traj_size = traj_info.groupby("traj_id_other")["time"].transform("size")
        traj_info["n_obs"] = (traj_info.groupby("traj_id_other").cumcount() + 1)
        traj_info["n_fut"] = traj_size - traj_info["n_obs"]
        ship_info.columns = [f"{c}_other" for c in ship_info.columns]

        edges = edges.merge(traj_info, on=["time", "mmsi_other"], how="inner")
        edges = edges.merge(ship_info, on=["traj_id_other"], how="inner")

        edges = edges[(edges["n_obs"] >= self.min_valid_window) & (edges["n_fut"] >= self.min_valid_window)]

        traj_id_in_frame = get_in_frame_dict(edges)

        edges["dist"] = np.log1p(edges["dist"]) / np.log1p(max_edge_dist)
        edges["rel_speed"] = edges["rel_speed"] / 40
        edges["tcpa"] = np.log1p(edges["tcpa"]) / np.log1p(3600)
        edges["dcpa"] = np.log1p(edges["dcpa"]) / np.log1p(2000)

        edges, edge_feat_cols = norm_deg(edges, 'course_diff', edge_feat_cols)
        edges, edge_feat_cols = norm_deg(edges, 'rel_bearing', edge_feat_cols)
        
        edge_feat_cols += static_feat_cols + [f"{c}_other" for c in static_feat_cols]
        self.num_edge_feats = len(edge_feat_cols)
        edges = edges.fillna(0)[["time", "traj_id", "traj_id_other"] + edge_feat_cols].copy()

        for traj_id, traj in tqdm(nodes.groupby("traj_id"), desc="Nodes"):
            for cur_t in traj.index[self.min_valid_window:-self.min_valid_window]:
                neighbors = traj_id_in_frame.get((cur_t, traj_id), [])
                self.items.append((cur_t, traj_id, neighbors))

            feat = traj[["rel_x", "rel_y"] + node_feat_cols].to_numpy(dtype=np.float32)
            padded_feat = np.pad(feat, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
            self.feat_map[traj_id] = padded_feat

        for row in tqdm(edges.itertuples(index=False, name=None), total=len(edges), desc="Edges"):
            time, traj_id, traj_id_other = row[:3]
            self.edge_map[(time, traj_id)][traj_id_other] = row[3:]


    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        cur_t, target_id, neighbors = self.items[index]

        node_ids = [target_id] + neighbors

        obs_feat_list = []
        obs_raw_list = []
        obs_pos_list = []
        obs_mask_list = []
        static_list = []

        for traj_id in node_ids:
            t0 = self.t0[traj_id]
            pos = self.pos_map[traj_id]
            feat = self.feat_map[traj_id]
            raw = self.raw_map[traj_id]

            t = self.l_pad + cur_t-t0

            obs_feat_list.append(feat[t-self.obs_len:t])
            obs_raw_list.append(raw[t-self.obs_len:t])
            obs_pos_list.append(pos[t-self.obs_len:t])
            obs_mask_list.append(np.arange(t - self.obs_len, t) >= self.l_pad)

            static_list.append(self.static_map[traj_id])

        t0 = self.t0[target_id]
        pos = self.pos_map[target_id]
        feat = self.feat_map[target_id]
        raw = self.raw_map[target_id]

        t = self.l_pad + cur_t-t0

        valid_end = len(pos) - self.r_pad
        fut_pos = [pos[t:t+self.pred_len]]
        fut_rel_pos = [feat[t:t+self.pred_len, :2]]
        fut_mask = [np.arange(t, t + self.pred_len) < valid_end]
        fin_pos = [pos[-(self.r_pad + 1)]]
        fin_pos_mask = [self.fin_pos_mask[target_id]]

        y_heatmap = []
        if self.pred_len == 0:
            y_heatmap = self.rasterize_occupancy(torch.from_numpy(pos[t:valid_end]))

        edge_index = [[],[]]
        edge_attr = []

        target_idx = 0
        edges = self.edge_map.get((cur_t, target_id))
        if edges:
            for source_idx, source_traj_id in enumerate(node_ids):
                edge = edges.get(source_traj_id)
                if not edge:
                    continue
                edge_attr.append(edge)
                edge_index[0].append(source_idx)
                edge_index[1].append(target_idx)

        is_ego = torch.zeros(len(node_ids), dtype=torch.bool)
        is_ego[0] = True

        if len(edge_attr) == 0:
            edge_attr = torch.empty((0, self.num_edge_feats), dtype=torch.float)
        else:
            edge_attr = torch.from_numpy(np.asarray(edge_attr)).float()

        data = Data(
            x_pos=torch.from_numpy(np.asarray(obs_pos_list)).float(),
            x=torch.from_numpy(np.asarray(obs_feat_list)).float(),
            x_raw=torch.from_numpy(np.asarray(obs_raw_list)).float(),
            x_mask=torch.from_numpy(np.asarray(obs_mask_list)).bool(),
            static=torch.from_numpy(np.asarray(static_list)).float(),

            edge_index=torch.from_numpy(np.asarray(edge_index)).long(),
            edge_attr=edge_attr,

            y_pos=torch.from_numpy(np.asarray(fut_pos)).float(),
            y_rel_pos=torch.from_numpy(np.asarray(fut_rel_pos)).float(),
            y_mask=torch.from_numpy(np.asarray(fut_mask)).bool(),
            fin_pos=torch.from_numpy(np.asarray(fin_pos)).float(),
            fin_pos_mask=torch.from_numpy(np.asarray(fin_pos_mask)).bool(),
            is_ego=is_ego,
            y_heatmap=y_heatmap,
        )

        return data
    
    def _pad(self, x):
        return np.pad(x, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
    
    def rasterize_occupancy(self, fut_pos):
        """
        Renders multiple future positions into a single occupancy grid.
        """
        x_bins, y_bins, *_ = RASTER.get_total_grid_sizes()

        x_idx = torch.floor((fut_pos[:, 0] - RASTER.x_min) / RASTER.pos_res).to(torch.int64)
        y_idx = torch.floor((fut_pos[:, 1] - RASTER.y_min) / RASTER.pos_res).to(torch.int64)
        
        x_idx = x_idx.clamp(0, x_bins - 1)
        y_idx = y_idx.clamp(0, y_bins - 1)

        grid = torch.zeros((x_bins, y_bins))
        
        grid[x_idx, y_idx] = 1.0
        grid = grid / (grid.sum() + 1e-8)

        return grid