import pandas as pd
import numpy as np

from typing import Iterable
from movingpandas import Trajectory, TrajectoryCollection
from scipy.interpolate import PchipInterpolator
from preprocessing.utils.df_transformer import to_GeoDataFrame


def interpolate(trajectories: Iterable[Trajectory], interp_cols, ff_cols):
    interpolated_trajectories = []

    for traj in trajectories:
        df = traj.df.sort_values(by="timestamp").reset_index()

        t_obs = df["timestamp"].astype("datetime64[s]").astype(int).to_numpy()
        y_obs = df[interp_cols].to_numpy()

        interpolator = PchipInterpolator(t_obs, y_obs)

        t_grid, t_nearest_diff = time_grid_and_proximity(t_obs)

        if len(t_grid) < 2:
            continue

        y_grid = interpolator(t_grid)

        interp = {
            "timestamp": pd.to_datetime(t_grid, unit="s"),
            "mmsi": traj.df["mmsi"].max(),
            "traj_id": traj.id,
            "time_diff": t_nearest_diff.astype(int),
        }
        for i, col in enumerate(interp_cols):
            interp[col] = y_grid[:, i].round(6)

        interp_df = pd.DataFrame(interp)
        ff_df = df[["timestamp"] + ff_cols]
        df = pd.merge_asof(interp_df, ff_df, on="timestamp")

        new_traj = Trajectory(
            to_GeoDataFrame(df, index="timestamp"),
            traj_id=traj.id,
            obj_id=traj.df["mmsi"].max(),
            t="timestamp",
        )

        if len(new_traj.df) >= 2:
            interpolated_trajectories.append(new_traj)
    trajectories = TrajectoryCollection(
        interpolated_trajectories,
        traj_id_col="traj_id",
        obj_id_col="mmsi",
        t="timestamp",
    )
    return trajectories


def time_grid_and_proximity(t, stepsize=10):
    t_start = int(np.ceil(t[0] / stepsize) * stepsize)
    t_end = int(np.floor(t[-1] / stepsize) * stepsize)
    t_grid = np.arange(t_start, t_end + 1, stepsize)

    t_proximity = np.min(np.abs(t[:, None] - t_grid), axis=0)
    return t_grid, t_proximity
