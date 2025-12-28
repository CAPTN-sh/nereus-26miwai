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
from sceen_loader.normalizer import normalize

def seq_collate(data):
    (
        obs_feat_seq,
        obs_pos_seq,
        obs_pos_rel_seq,
        fut_pos_seq,
        fut_pos_rel_seq,
        train_mask_seq,
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
    fut_pos = torch.cat(fut_pos_seq, dim=0)
    fut_pos_rel = torch.cat(fut_pos_rel_seq, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    train_mask = torch.cat([m.to(torch.bool) for m in train_mask_seq], dim=0)

    return (
        obs_feat,
        obs_pos,
        obs_pos_rel,
        fut_pos,
        fut_pos_rel,
        seq_start_end,
        train_mask,
    )


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
    latlon_cols = [col for col in df.columns if "lat" in col or "lon" in col]
    if len(latlon_cols) % 2 != 0:
        raise ValueError(f"Uneven number of lat/lon columns: {latlon_cols}")

    # TODO get CRS from config
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:25832"), always_xy=True
    )
    for i in range(0, len(latlon_cols), 2):
        lon_col = latlon_cols[i] if "lon" in latlon_cols[i] else latlon_cols[i + 1]
        lat_col = latlon_cols[i] if "lat" in latlon_cols[i] else latlon_cols[i + 1]
        df[lon_col], df[lat_col] = transformer.transform(
            df[lon_col].values, df[lat_col].values
        )
    df.columns = [col.replace("lon", "x").replace("lat", "y") for col in df.columns]
    return df


class SceenTrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        feat_cols=[],
        obs_len=24,
        pred_len=24,
        exclude_ship_types=list(range(0, 40)),
        force_rebuild=True,
        normalizer_path = None,
    ):
        super(SceenTrajectoryDataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        # load from cache
        folder_name = nodes_path.parent.parent.name + nodes_path.parent.name
        feature_name = "".join([f[0] for f in feat_cols])
        cache_name = (
            f"sceen_loader_{min_date.date()}_{max_date.date()}_"
            f"{obs_len}_{pred_len}__{feature_name}_{folder_name}"
        )
        cache_path = Path("data/cache") / f"{cache_name}.pkl"
        os.makedirs(cache_path.parent, exist_ok=True)

        if cache_path.exists() and not force_rebuild:
            print(f"Loading dataset from cache: {cache_path}")
            self.items, self.data, self.feature_cols = joblib.load(cache_path)
            return

        print(f"Cache not found: {cache_path}. Processing dataset...")

        self.items = []
        self.data = {}
        self.feature_cols = feat_cols

        nodes = pd.read_parquet(nodes_path)

        step_size = get_interp_step_size(nodes)
        nodes = process_time(nodes, min_date, max_date, step_size)

        if nodes.empty:
            raise ValueError("There are no values within the given time range.")

        # TODO config
        ship_db_path = Path("/data/projects/ais/data/ship_db/ship_db.parquet")
        ship_db = pd.read_parquet(ship_db_path)
        nodes = nodes.reset_index().merge(ship_db, on="mmsi")

        exclude_mmsi = set(
            ship_db[ship_db["ship_type"].isin(exclude_ship_types)]["mmsi"]
        )

        edges = pd.read_parquet(edges_path)
        edges = edges.reset_index().drop(columns=["timestamp"])
        edges = process_time(edges, min_date, max_date, step_size)
        risk_mmsi = edges.sort_values(
            ["collision_risk", "dist"], ascending=[False, True]
        ).drop_duplicates(subset=["time", "mmsi"], keep="first")

        # TODO cleanup of columns
        nodes = nodes.fillna(0)

        nodes = nodes.merge(risk_mmsi, on=["time", "mmsi"])
        nodes = nodes.set_index("time").sort_index()

        nodes["length"] = nodes['to_bow'] + nodes['to_stern']
        nodes["width"] = nodes['to_port'] + nodes['to_starboard']

        nodes["density"] = nodes["density_all"]
        for group in ["sailing", "cargo", "passenger"]:
            mask = nodes["ship_group"] == group
            nodes.loc[mask, "density"] = nodes.loc[mask, f"density_{group}"]

        for col in ['density_all', 'density_sailing', 'density_cargo',
            'density_other', 'density_passenger', 'density']:
            nodes[col] = np.log1p(nodes[col])

        nodes = nodes[["mmsi", "lat", "lon"] + self.feature_cols]
        nodes = normalize(nodes, normalizer_path, cache_name)
        self.feature_cols = nodes.columns

        nodes = cords_to_meters(nodes)
        self.feature_cols = [col.replace("lon", "x") for col in self.feature_cols]
        self.feature_cols = [col.replace("lat", "y") for col in self.feature_cols]


        nodes = nodes.sort_index()
        for cur_t in tqdm(range(nodes.index[0], nodes.index[-1])):
            self._add_items_at_t(nodes, cur_t, exclude_mmsi)

        # build per-mmsi DataFrames (then convert to numpy)
        for mmsi, group in tqdm(nodes.groupby("mmsi")):
            df = add_filled_gap_steps(group, self.pred_len)
            df = add_rel_pos(df)
            self.data[mmsi] = df

        # finally convert self.data to numpy representation
        self._to_numpy()

        print(f"Saving dataset to cache: {cache_path}")
        joblib.dump((self.items, self.data, self.feature_cols), cache_path)


    def _add_items_at_t(self, nodes, cur_t, exclude_mmsi):
        nodes_t = nodes.loc[cur_t - self.obs_len : cur_t + self.pred_len - 1]
        if nodes_t.empty:
            return

        traj_len = nodes_t.groupby("mmsi").size().to_dict()
        traj_len = {mmsi: l for mmsi, l in traj_len.items() if l >= self.pred_len}
        sceen_mmsi = list(traj_len.keys())

        train_mask = [
            (len_i == self.obs_len + self.pred_len) and (mmsi not in exclude_mmsi)
            for mmsi, len_i in traj_len.items()
        ]

        if not any(train_mask):
            return

        self.items.append((cur_t, sceen_mmsi, train_mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        cur_t, sceen_mmsi, train_mask = self.items[index]

        obs_feat = []
        obs_pos = []
        obs_pos_rel = []
        fut_pos = []
        fut_pos_rel = []

        for mmsi in sceen_mmsi:
            entry = self.data[mmsi]
            idx_map = entry["time_to_idx"]

            # Build index lists from actual time stamps, so gaps are fine
            try:
                obs_idx = [
                    idx_map[cur_t - self.obs_len + k] for k in range(self.obs_len)
                ]
                fut_idx = [
                    idx_map[cur_t + k] for k in range(self.pred_len)
                ]
            except KeyError as e:
                # This *shouldn't* happen because _add_items_at_t only keeps
                # complete trajectories. If it does, fail loudly so you can inspect.
                raise ValueError(
                    f"Missing time {e.args[0]} for mmsi {mmsi} at cur_t={cur_t}"
                )

            obs_feat.append(entry["feat"][obs_idx, :])
            obs_pos.append(entry["pos"][obs_idx, :])
            obs_pos_rel.append(entry["pos_rel"][obs_idx, :])
            fut_pos.append(entry["pos"][fut_idx, :])
            fut_pos_rel.append(entry["pos_rel"][fut_idx, :])

        # Now all lists have consistent shapes: [N, obs_len/pred_len, C]
        obs_feat = self._to_tensor(obs_feat)
        obs_pos = self._to_tensor(obs_pos)
        obs_pos_rel = self._to_tensor(obs_pos_rel)
        fut_pos = self._to_tensor(fut_pos)
        fut_pos_rel = self._to_tensor(fut_pos_rel)
        train_mask = torch.as_tensor(train_mask, dtype=torch.bool)

        return obs_feat, obs_pos, obs_pos_rel, fut_pos, fut_pos_rel, train_mask

    def _to_tensor(self, traj):
        # traj: list of numpy arrays [T, C] -> tensor [N, C, T]
        return torch.tensor(np.stack(traj, axis=0)).permute(0, 2, 1).float()

    def _to_numpy(self):
        """
        Convert self.data[mmsi] from DataFrames into compact numpy arrays
        plus a time->index lookup, so that we can index by actual time
        labels even if the index has gaps.

        After this:
            self.data[mmsi] = {
                "times": np.ndarray,        # [T]
                "time_to_idx": dict[int,int],
                "feat": np.ndarray,         # [T, F]
                "pos": np.ndarray,          # [T, 2]  (x, y)
                "pos_rel": np.ndarray,      # [T, 2]  (rel_x, rel_y)
            }
        """
        for mmsi, df in list(self.data.items()):
            # Already converted?
            if isinstance(df, dict) and "feat" in df:
                continue

            df = df.sort_index()
            times = df.index.to_numpy()

            feat = df[self.feature_cols].to_numpy(dtype=np.float32)
            pos = df[["x", "y"]].to_numpy(dtype=np.float32)
            pos_rel = df[["rel_x", "rel_y"]].to_numpy(dtype=np.float32)

            # Map actual time label -> row index
            # int(...) makes keys plain Python ints (cur_t from range(...) is also int)
            time_to_idx = {int(t): i for i, t in enumerate(times)}

            self.data[mmsi] = {
                "times": times,
                "time_to_idx": time_to_idx,
                "feat": feat,
                "pos": pos,
                "pos_rel": pos_rel,
            }


