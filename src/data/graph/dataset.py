import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data, Dataset

from data.graph.utils.heat_map import rasterize_occupancy
from data.graph.utils.normalize import normalize_ship_db, normalize_nodes, normalize_edges
from data.graph.utils.transform import process_time, get_in_frame_dict, cords_to_meters

from utils.config import SHIP_DB_PATH, STEPS_PER_MINUTE

class GraphTrajectoryDataset(Dataset):
    """
    Graph-based AIS trajectory dataset that constructs spatio-temporal samples
    with dynamic, static, and interaction features for vessel prediction.
    Each item represents one target vessel and its neighbours at a specific timeframe.
    """

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
        ship_group = "all",
    ):
        super(GraphTrajectoryDataset, self).__init__()

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

        #### NODES ####
        nodes = pd.read_parquet(nodes_path)
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

        nodes, node_feat_cols = normalize_nodes(nodes, node_feat_cols)
        self.num_node_feats = 2 + len(node_feat_cols)

        #### STATIC ####
        ship_db = pd.read_parquet(SHIP_DB_PATH)
        traj_id_map = nodes[["mmsi", "traj_id"]].drop_duplicates()
        ship_db = traj_id_map.merge(ship_db, on="mmsi", how="left")

        ship_group_df = ship_db[["traj_id", "ship_group"]].fillna(0.0).set_index("traj_id").copy()

        ship_db, static_feat_cols = normalize_ship_db(ship_db, static_feat_cols)
        ship_db = ship_db[["traj_id"] + static_feat_cols].fillna(0.0).copy()
        self.num_static_feats = len(static_feat_cols)

        self.static_map = {
            traj_id: row.to_numpy(dtype=np.float32)
            for traj_id, row in ship_db.set_index("traj_id")[static_feat_cols].iterrows()
        }

        #### EDGES ####
        if not max_edge_dist:
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

        edges, edge_feat_cols = normalize_edges(edges, edge_feat_cols, static_feat_cols, max_edge_dist)
        self.num_edge_feats = len(edge_feat_cols)
        edges = edges.fillna(0)[["time", "traj_id", "traj_id_other"] + edge_feat_cols].copy()

        for traj_id, traj in tqdm(nodes.groupby("traj_id"), desc="Nodes"):
            if (ship_group == "all") or (ship_group == ship_group_df.loc[traj_id, "ship_group"]):
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

        all_fut_pos_list = []
        all_fut_mask_list = []

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

            valid_end = len(pos) - self.r_pad
            all_fut_pos_list.append([pos[t:t+self.pred_len]])
            all_fut_mask_list.append(np.arange(t, t + self.pred_len) < valid_end)

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
            y_heatmap = rasterize_occupancy(torch.from_numpy(pos[t:valid_end]))

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

            target_id = target_id,
            y_all = torch.from_numpy(np.asarray(all_fut_pos_list)).float(),
            y_all_mask = torch.from_numpy(np.asarray(all_fut_mask_list)).float(),
        )

        return data
    
    def _pad(self, x):
        return np.pad(x, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')
