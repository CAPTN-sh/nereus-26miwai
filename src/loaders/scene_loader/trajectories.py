from datetime import timedelta
from pathlib import Path
import joblib
import os
import numpy as np
import pandas as pd
import pyproj
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from loaders.scene_loader.normalizer import normalize

from utils.config import SHIP_DB_PATH, STEPS_PER_MINUTE, STEP_SIZE

def seq_collate(data):
    (
        obs_feat_seq,
        obs_pos_seq,
        obs_pos_rel_seq,
        obs_mask_seq,
        fut_pos_seq,
        fut_pos_rel_seq,
        fut_mask_seq,
    ) = zip(*data)

    _len = [len(seq) for seq in obs_pos_seq]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    # [N_total, C, T] (batch, channels, time)
    obs_feat = torch.cat(obs_feat_seq, dim=0)
    obs_pos = torch.cat(obs_pos_seq, dim=0)
    obs_pos_rel = torch.cat(obs_pos_rel_seq, dim=0)
    obs_mask = torch.cat(obs_mask_seq, dim=0)
    fut_pos = torch.cat(fut_pos_seq, dim=0)
    fut_pos_rel = torch.cat(fut_pos_rel_seq, dim=0)
    fut_mask = torch.cat(fut_mask_seq, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    return (
        obs_feat,
        obs_pos,
        obs_pos_rel,
        obs_mask,
        fut_pos,
        fut_pos_rel,
        fut_mask,
        seq_start_end,
    )


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
    mmsis_in_frame = (
        edges.sort_values(by=["time", "mmsi", "dist"])
        .groupby(["time", "mmsi"])["mmsi_other"]
        .apply(list)
        .to_dict()
    )
    return mmsis_in_frame


def cords_to_meters(df: pd.DataFrame):
    # TODO get CRS from config
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:25832"), always_xy=True
    )
    df["x"], df["y"] = transformer.transform(df["lon"].values, df["lat"].values)
    df = df.drop(columns = ["lon", "lat"])
    return df


class SceneTrajectoryDataset(Dataset):
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
        force_rebuild=False
    ):
        super(SceneTrajectoryDataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.min_valid_window = min_len_in_minutes * STEPS_PER_MINUTE

        self.l_pad = obs_len
        self.r_pad = pred_len

        # load from cache
        cache_name = f"sl_{min_date.date()}_{max_date.date()}_{obs_len}_{pred_len}_"
        cache_name += "".join([f[0] for f in feat_cols])

        cache_dir = Path("data/cache") / nodes_path.parent.parent.name / nodes_path.parent.name / flag / cache_name
        meta_path = cache_dir / "meta.pkl"
        feat_path = cache_dir / "feat.dat"
        pos_path = cache_dir / "pos.dat"
        pos_rel_path = cache_dir / "pos_rel.dat"

        if cache_dir.exists() and not force_rebuild:
            print(f"Loading dataset from cache: {cache_dir}")
            with open(meta_path, "rb") as f:
                meta = joblib.load(f)
            self.items = meta["items"]
            self.traj_id_to_idx = meta["traj_id_to_idx"]
            self.t0 = meta["t0"]
            self.lengths = meta["lengths"]
            self.offsets = meta["offsets"]

            self.feat = np.memmap(feat_path, dtype="float32", mode="r", shape=meta["feat_shape"])
            self.pos = np.memmap(pos_path, dtype="float32", mode="r", shape=meta["pos_shape"])
            self.pos_rel = np.memmap(pos_rel_path, dtype="float32", mode="r", shape=meta["pos_rel_shape"])
            
            self.items = [(cur_t, traj_ids) for (cur_t, traj_ids) in self.items if len(traj_ids) <= 100]
            return
        
        print(f"Cache not found: {cache_dir}")
        print("Or force rebuild...")

        self.items = []
        self.data = {}
        self.traj_id_to_idx = {}
        self.t0 = []
        self.feat = []
        self.pos = []
        self.pos_rel = []
        self.lengths = []
        self.feature_cols = feat_cols.copy()

        nodes = pd.read_parquet(nodes_path)
        nodes = process_time(nodes, min_date, max_date)

        if nodes.empty:
            raise ValueError("There are no values within the given time range.")

        ship_db = pd.read_parquet(SHIP_DB_PATH)
        nodes = nodes.reset_index().merge(ship_db, on="mmsi")

        # edges (see bottom)

        nodes = nodes.fillna(0)
        nodes = nodes.set_index("time").sort_index()

        # add features
        nodes["hour_of_day"] = nodes["timestamp"].dt.hour
        
        nodes["length"] = nodes['to_bow'] + nodes['to_stern']
        nodes["width"] = nodes['to_port'] + nodes['to_starboard']

        nodes = nodes[["traj_id", "lat", "lon"] + self.feature_cols]
        nodes = cords_to_meters(nodes)
        traj_g = nodes.groupby("traj_id")
        nodes[["rel_x", "rel_y"]] = traj_g[["x", "y"]].diff().fillna(0) / 100.0
        nodes, self.feature_cols = normalize(nodes, self.feature_cols)

        nodes["n_from_start"] = traj_g.cumcount()
        nodes["n_to_end"] = traj_g.transform("size") - 1 - nodes["n_from_start"]
        
        max_ships_per_scene = 0
        for cur_t, nodes_t in tqdm(nodes.groupby(level=0)):
            agent_mask = (nodes_t["n_from_start"] >= self.min_valid_window) & (nodes_t["n_to_end"] >= self.min_valid_window)
            valid_traj_ids = nodes_t.loc[agent_mask, "traj_id"].tolist()
            if len(valid_traj_ids) > 0:
                self.items.append((cur_t, valid_traj_ids))
                max_ships_per_scene = max(max_ships_per_scene, len(valid_traj_ids))
        
        print("max_ships_per_scene:", max_ships_per_scene)

        for idx, (traj_id, df) in enumerate(tqdm(nodes.groupby("traj_id"))):
            self.traj_id_to_idx[traj_id] = idx
            self.t0.append(df.index[0])

            feat_vec = self._pad(df[self.feature_cols].to_numpy(dtype=np.float32))
            self.feat.append(feat_vec)
            self.pos.append(self._pad(df[["x", "y"]].to_numpy(dtype=np.float32)))
            self.pos_rel.append(self._pad(df[["rel_x", "rel_y"]].to_numpy(dtype=np.float32)))

            self.lengths.append(len(feat_vec))

        print(f"Saving dataset to cache: {cache_dir}")
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        self.offsets = np.zeros(len(self.lengths) + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.lengths)
        
        feat_all = np.concatenate(self.feat, axis=0)
        pos_all = np.concatenate(self.pos, axis=0)
        pos_rel_all = np.concatenate(self.pos_rel, axis=0)
        
        cache_dir.mkdir(parents=True, exist_ok=True)

        def flush_array(array, path):
            mm = np.memmap(path, dtype="float32", mode="w+", shape=array.shape)
            mm[:] = array
            mm.flush()

        flush_array(feat_all, feat_path)
        flush_array(pos_all, pos_path)
        flush_array(pos_rel_all, pos_rel_path)

        meta = {
            "items": self.items,
            "traj_id_to_idx": self.traj_id_to_idx,
            "offsets": self.offsets,
            "lengths": self.lengths,
            "t0": np.asarray(self.t0, dtype=np.int64),
            "feat_shape": feat_all.shape,
            "pos_shape": pos_all.shape,
            "pos_rel_shape": pos_rel_all.shape,
        }

        with open(meta_path, "wb") as f:
            joblib.dump(meta, f, protocol=5)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        cur_t, traj_ids = self.items[index]

        obs_feat = []
        obs_pos = []
        obs_pos_rel = []
        obs_mask = []
        fut_pos = []
        fut_pos_rel = []
        fut_mask = []

        for traj_id in traj_ids:
            idx = self.traj_id_to_idx[traj_id]
            start = self.offsets[idx]
            end = self.offsets[idx + 1]

            t_0 = self.t0[idx]
            feat = self.feat[start:end]
            abs_pos = self.pos[start:end]
            rel_pos = self.pos_rel[start:end]

            #t_0, feat, abs_pos, rel_pos = self.data[traj_id]
            cur_idx = self.l_pad + cur_t - t_0

            obs_feat.append(feat[cur_idx - self.obs_len : cur_idx])
            obs_pos.append(abs_pos[cur_idx - self.obs_len : cur_idx])
            obs_pos_rel.append(rel_pos[cur_idx - self.obs_len : cur_idx])

            fut_pos.append(abs_pos[cur_idx : cur_idx + self.pred_len])
            fut_pos_rel.append(rel_pos[cur_idx : cur_idx + self.pred_len])

            obs_mask.append((np.arange(cur_idx - self.obs_len, cur_idx) >= self.l_pad)[:, None])
            fut_mask.append((np.arange(cur_idx, cur_idx + self.pred_len) < (len(abs_pos) - self.r_pad))[:, None])
        
        # Now all lists have consistent shapes: [N, obs_len/pred_len, C]
        obs_feat = self._to_tensor(obs_feat)
        obs_pos = self._to_tensor(obs_pos)
        obs_pos_rel = self._to_tensor(obs_pos_rel)
        obs_mask= self._to_tensor(obs_mask, dtype=torch.bool)
        fut_pos = self._to_tensor(fut_pos)
        fut_pos_rel = self._to_tensor(fut_pos_rel)
        fut_mask= self._to_tensor(fut_mask, dtype=torch.bool)

        return obs_feat, obs_pos, obs_pos_rel, obs_mask, fut_pos, fut_pos_rel, fut_mask

    def _to_tensor(self, traj, dtype=torch.float32):
        # traj: list of numpy arrays [T, C] -> tensor [N, C, T]
        return torch.from_numpy(np.stack(traj, axis=0)).to(dtype).permute(0, 2, 1)
    
    def _pad(self, x):
        return np.pad(x, ((self.l_pad, self.r_pad), (0, 0)), mode='constant')


"""
        if False:
            edges = pd.read_parquet(edges_path)
            edges = edges.reset_index().drop(columns=["timestamp"])
            edges = process_time(edges, min_date, max_date, step_size)

            risk_mmsi = edges.sort_values(
                ["collision_risk", "dist"], ascending=[False, True]
            ).drop_duplicates(subset=["time", "mmsi"], keep="first")
            nodes = nodes.merge(risk_mmsi, on=["time", "mmsi"])
"""