import pandas as pd
import numpy as np
from pandas import DataFrame
from geopandas import GeoDataFrame
from typing import Iterable
from movingpandas import Trajectory, TrajectoryCollection
from scipy.interpolate import PchipInterpolator
from preprocessing.utils.df_helper import to_GeoDataFrame
from typing import List, Tuple


def interpolate(
    trajectories: Iterable[Trajectory], interp_cols: List[str], ff_cols: List[str]
) -> Iterable[Trajectory]:
    """
    Interpolates the trajectory to aligne the timestamps of all trajectories
    to every full 10 seconds. Also the time diff to the clostest real timestamp is stored.

    Args:
        trajectories (Iterable[Trajectory]): Traj to apply the function on.
        interp_cols (List[str]): Columns to interpolate.
        ff_cols (List[str]): Columns to feed forward to the new timestamps.

    Return:
        Iterable[Trajectory]: Interpolated trajectories with new columns.
    """
    interpolated_trajectories = []

    for traj in trajectories:
        df = interpolate_df(traj.df, interp_cols, ff_cols)

        if df is None:
            continue

        new_traj = Trajectory(
            to_GeoDataFrame(df, index="timestamp"),
            traj_id=traj.id,
            obj_id=traj.df["mmsi"].max(),
            t="timestamp",
        )
        interpolated_trajectories.append(new_traj)

    trajectories = TrajectoryCollection(
        interpolated_trajectories,
        traj_id_col="traj_id",
        obj_id_col="mmsi",
        t="timestamp",
    )
    return trajectories


def interpolate_df(
    df: GeoDataFrame, interp_cols: List[str], ff_cols: List[str]
) -> DataFrame:
    """
    Interpolates the given GeoDataFrame.

    Args:
        df (GeoDataFrame): GeoDataFrame to apply the function on.
        interp_cols (List[str]): Columns to interpolate.
        ff_cols (List[str]): Columns to feed forward to the new timestamps.

    Return:
        DataFrame: Interpolated DataFrame with new columns.
    """
    df = df.sort_values(by="timestamp").reset_index()

    t_obs = df["timestamp"].astype("datetime64[s]").astype(int).to_numpy()
    y_obs = df[interp_cols].to_numpy()

    interpolator = PchipInterpolator(t_obs, y_obs)
    t_grid, t_nearest_diff = _time_grid_and_proximity(t_obs)

    if len(t_grid) < 2:
        return

    interp = {
        "timestamp": pd.to_datetime(t_grid, unit="s"),
        "mmsi": df["mmsi"].iloc[0],
        "traj_id": df["traj_id"].iloc[0],
        "time_diff": t_nearest_diff.astype(int),
    }

    y_grid = interpolator(t_grid)
    for i, col in enumerate(interp_cols):
        interp[col] = y_grid[:, i].round(6)

    ff_df = df[["timestamp"] + ff_cols]
    df = pd.merge_asof(DataFrame(interp), ff_df, on="timestamp")
    return df


def _time_grid_and_proximity(
    t: np.ndarray, stepsize=10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alinge the timestamps to a time grid with the given stepsize.

    Args:
        t: Original timestamps.
        stepsize: Stepsize of the grid.

    Returns:
        np.ndarray: Time grid within the min and max of t.
        np.ndarray: The min dist to the original timestamps.
    """

    t_start = int(np.ceil(t[0] / stepsize) * stepsize)
    t_end = int(np.floor(t[-1] / stepsize) * stepsize)
    t_grid = np.arange(t_start, t_end + 1, stepsize)

    t_proximity = np.min(np.abs(t[:, None] - t_grid), axis=0)
    return t_grid, t_proximity
