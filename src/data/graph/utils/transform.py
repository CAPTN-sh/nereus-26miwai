import pandas as pd
import pyproj
from datetime import timedelta

from utils.config import STEP_SIZE, METER_CRS, DEFAULT_CRS

def process_time(df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    """Filter by date range and convert timestamps to discrete time steps."""
    df = df[df["timestamp"].between(min_date, max_date + timedelta(days=1))].copy().reset_index(drop=True)
    df["time"] = df["timestamp"].astype("datetime64[s]").astype("int64") // STEP_SIZE
    return df.set_index("time").sort_index()

def get_in_frame_dict(edges):
    """Map (time, traj_id) → list of neighboring trajectory IDs."""
    traj_id_in_frame = (
        edges.sort_values(by=["time", "traj_id", "dist"])
        .groupby(["time", "traj_id"])["traj_id_other"]
        .apply(list)
        .to_dict()
    )
    return traj_id_in_frame

def cords_to_meters(df: pd.DataFrame):
    """Project geographic coordinates (lat/lon) to metric CRS (x, y)."""
    transformer = pyproj.Transformer.from_crs(pyproj.CRS(DEFAULT_CRS), pyproj.CRS(METER_CRS), always_xy=True)
    df["x"], df["y"] = transformer.transform(df["lon"].values, df["lat"].values)
    df = df.drop(columns = ["lon", "lat"])
    return df