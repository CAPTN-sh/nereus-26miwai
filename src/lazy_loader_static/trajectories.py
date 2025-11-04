from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def seq_collate():
    return


def get_interp_step_size(df: pd.DataFrame):
    first_traj = df.loc[df["traj_id"] == df.iloc[0]["traj_id"]].copy()
    ts = first_traj.sort_values("timestamp")["timestamp"]
    step_size = int((ts.iloc[1] - ts.iloc[0]).total_seconds())
    return step_size


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


class LazyTrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        feat_cols=[],
        slice_len=128,
    ):
        """
        Args:
            TODO
        """
        super(LazyTrajectoryDataset, self).__init__()
        self.items = []
        self.data = {}
        self.slice_len = slice_len
        self.feature_cols = feat_cols

        nodes = pd.read_parquet(nodes_path)

        step_size = get_interp_step_size(nodes)
        nodes = process_time(nodes, min_date, max_date, step_size)

        if nodes.empty:
            raise ValueError("There are no values within the given time range.")

        # TODO config
        # ship_db = pd.read_parquet(Path("data/ais/ship_db/ship_db.parquet"))
        # nodes = pd.merge(nodes, ship_db, how="left", on="mmsi")

        nodes = nodes[["mmsi", "lat", "lon"] + self.feature_cols]
        nodes = cords_to_meters(nodes)
        self.feature_cols = [
            col.replace("lon", "x").replace("lat", "y") for col in self.feature_cols
        ]

        edges = pd.read_parquet(edges_path)
        edges = process_time(edges, min_date, max_date, step_size)

        nodes_t = nodes.reset_index().rename(
            columns={nodes.index.name or "index": "time"}
        )
        edges_t = edges.reset_index().rename(
            columns={edges.index.name or "index": "time"}
        )

        # sector labeling (adjust bounds if your convention differs)
        bear = edges_t["rel_bearing"]
        edges_t["sector"] = np.select(
            [
                (bear >= 355) | (bear <= 5),  # bow (head on)
                (bear > 5) & (bear <= 112.5),  # port (crossing - give way)
                (bear >= 247.5) & (bear < 355),  # starboard (crossing - stand on)
            ],
            ["bow", "port", "starboard"],
            default="stern",  # overtaking
        )

        def fwd_min_counter(g: pd.DataFrame):
            idx = pd.to_timedelta(g["time"], unit="s")
            s = g.set_index(idx)["collision_risk"]
            # forward-looking window: reverse -> rolling('k units') -> reverse back
            horizon = pd.Timedelta(seconds=slice_len)
            out = s.iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1]
            g["total_collision_risk"] = out.values
            return g

        edges_t = edges_t.sort_values(["mmsi", "mmsi_other", "sector", "time"])
        edges_t = edges_t.groupby(
            ["mmsi", "mmsi_other", "sector"], group_keys=False
        ).apply(fwd_min_counter)

        idx = edges_t.groupby(["time", "mmsi", "sector"])[
            "total_collision_risk"
        ].idxmin()

        closest = edges_t.loc[idx, ["time", "mmsi", "sector", "mmsi_other"]]

        wide = closest.pivot_table(
            index=["time", "mmsi"],
            columns="sector",
            values=["mmsi_other"],
            aggfunc="first",
        )
        wide.columns = [f"{val}_{sec}" for (val, sec) in wide.columns]
        wide = wide.reset_index()

        others = nodes_t.merge(wide, on=["time", "mmsi"], how="left")
        others = others.set_index("time").sort_index()

        for cur_t in tqdm(range(nodes.index[0], nodes.index[-1])):
            self._add_items_at_t(nodes, cur_t, others)

        for mmsi, group in tqdm(nodes.groupby("mmsi")):
            df = add_filled_gap_steps(group, self.slice_len)
            self.data[mmsi] = df

    def _add_items_at_t(self, nodes, cur_t, others):
        nodes_t = nodes.loc[cur_t : cur_t + self.slice_len]
        if nodes_t.empty:
            return

        traj_len = nodes_t.groupby("mmsi").size().to_dict()
        traj_len = {mmsi: l for mmsi, l in traj_len.items() if l >= self.slice_len}
        valid_mmsi = list(traj_len.keys())

        others_t = others.loc[cur_t] if cur_t in others.index else None

        for mmsi in valid_mmsi:
            section = ["bow", "port", "starboard", "stern"]
            if others_t is None:
                other_mmmsi = [None, None, None, None]
            else:
                other_mmmsi = [others_t[f"mmsi_other_{s}"] for s in section]
            self.items.append((cur_t, mmsi, other_mmmsi))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        cur_t, cur_mmsi, others = self.items[index]

        obs_feat = [self.data[cur_mmsi][cur_t : cur_t + self.slice_len]]

        for mmsi in others:
            if mmsi is None:
                # TODO dummy data
                continue
            df = self.data[mmsi]
            obs_df = df.loc[cur_t : cur_t + self.slice_len]
            obs_feat.append(obs_df[["x", "y"] + self.feature_cols].values)

        obs_feat = self._to_tensor(obs_feat)

        return obs_feat

    def _to_tensor(self, traj):
        return torch.tensor(np.stack(traj, axis=0)).permute(0, 2, 1).float()
