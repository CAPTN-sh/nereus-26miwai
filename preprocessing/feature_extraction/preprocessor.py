import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.interpolate import CubicSpline


def process_traj(dfs):
    df_mmsi = [df for _, df in dfs[0].groupby("mmsi")]
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(_process_ship, df_mmsi), total=len(df_mmsi)))
    dfs[0] = pd.concat(results, ignore_index=True)
    return dfs


def _process_ship(df):
    steps = [
        filter_lat_lon,
        drop_duplicates,
        compute_speed,
        lambda d: drop_outlier(d, "sog", 30),
        calculate_rot,
        lambda d: drop_outlier(d, "rot", 200),
        # interpolate,
    ]

    for step in steps:
        df = step(df)
        if len(df) < 10:
            return pd.DataFrame(columns=df.columns)
    return df


def interpolate(df):
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().values
    t_new = np.arange(t[0], t[-1] + 1, 10)

    lat_new = CubicSpline(t, df["lat"])(t_new)
    lon_new = CubicSpline(t, df["lon"])(t_new)

    timestamp_new = df["timestamp"].iloc[0] + pd.to_timedelta(t_new, unit="s")
    df_interp = pd.DataFrame(
        {"timestamp": timestamp_new, "lat": lat_new, "lon": lon_new}
    )

    df_ffill = (
        df.set_index("timestamp")
        .reindex(timestamp_new, method="ffill")
        .reset_index(drop=True)
    )
    columns_to_ffill = df.columns.difference(["timestamp", "lat", "lon"])
    df_interp[columns_to_ffill] = df_ffill[columns_to_ffill]

    df_interp["real"] = df_interp["timestamp"].isin(df["timestamp"]).astype(bool)
    return df_interp


def filter_lat_lon(df):
    df = df[(df["lat"] > 54.3) & (df["lat"] < 54.8)]
    df = df[(df["lon"] > 9.8) & (df["lon"] < 11.0)]
    return df


def drop_duplicates(df):
    df["non_null_count"] = df.notnull().sum(axis=1)
    df = df.sort_values("non_null_count", ascending=False)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.drop(columns="non_null_count")
    return df


def distance(lat1, lon1, lat2, lon2):
    dx = (lon2 - lon1) * 60.0 * np.cos(np.radians((lat1 + lat2) / 2))
    dy = (lat2 - lat1) * 60.0
    return np.sqrt(dx**2 + dy**2)


def compute_speed(df: pd.DataFrame):
    df = df.sort_values("timestamp").copy()

    df["lat_prev"] = df["lat"].shift()
    df["lon_prev"] = df["lon"].shift()
    df["timestamp_prev"] = df["timestamp"].shift()

    dt = (df["timestamp"] - df["timestamp_prev"]).dt.total_seconds() / 3600
    dist = distance(df["lat_prev"], df["lon_prev"], df["lat"], df["lon"])
    df["sog"] = round(dist / dt, 1)
    df = df.reset_index(drop=True)
    if len(df) > 1:
        df.loc[0, "sog"] = df.loc[1, "sog"]

    df = df.drop(["lat_prev", "lon_prev", "timestamp_prev"], axis=1)
    return df


def calculate_rot(df):
    # Calculate time delta in minutes
    df["delta_time_min"] = df["timestamp"].diff().dt.total_seconds() / 60

    # Calculate change in course (normalize around 360)
    df["delta_course"] = df["course"].diff().mod(360)
    df["delta_course"] = np.where(
        (df["delta_course"] > 180) & df["delta_course"].notna(),
        df["delta_course"] - 360,
        df["delta_course"],
    )

    # Calculate ROT (°/min)
    df["rot"] = df["delta_course"] / df["delta_time_min"]

    df = df.drop(["delta_course", "delta_time_min"], axis=1)
    return df


def drop_outlier(df: pd.DataFrame, col, max_threshold):
    T = box_plot_threshold(df[col])
    T = min(T, max_threshold)
    if "outlier" in df.columns:
        df["outlier"] = df["outlier"] | (np.abs(df[col]) > T)
    else:
        df["outlier"] = np.abs(df[col]) > T
    return df


def box_plot_threshold(x):
    x = np.abs(x)
    x = x[x > 1]
    if len(x) == 0:
        return 1
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    IQR = q3 - q1
    T = q3 + 1.5 * IQR
    return T
