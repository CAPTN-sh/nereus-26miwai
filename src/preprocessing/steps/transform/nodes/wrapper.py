from movingpandas import Trajectory
from typing import Iterable
from preprocessing.utils.pipeline.function_inport import import_from_string
import re
from datetime import timedelta


def outlier(trajectories: Iterable[Trajectory], column, measures, threshold, drop=True):
    for traj in trajectories:
        values = traj.df[column].copy()

        for measure in measures:
            values = import_from_string(measure)(values)
        outliers = values > threshold

        if drop:
            traj.df = traj.df.loc[~outliers].copy()
        else:
            traj.df[f"{column}_outlier"] = outliers
    return trajectories


def split(trajectories: Iterable[Trajectory], cls, **args):
    splitter = import_from_string(cls)(trajectories)

    pattern = re.compile(r"timedelta_in_min_(\d+)")
    for key, value in args.items():
        args[key] = value
        if isinstance(value, str):
            match = pattern.match(value)
            if match:
                args[key] = timedelta(minutes=int(match.group(1)))
    trajectories = splitter.split(**args)
    return trajectories


def smooth(trajectories: Iterable[Trajectory], cls, **args):
    smoother = import_from_string(cls)(trajectories)
    trajectories = smoother.smooth(**args)
    return trajectories


def anchored_default(trajectories: Iterable[Trajectory], max_speed, default_values):
    for traj in trajectories:
        anchored = traj.df[traj.speed_col_name].max() <= max_speed
        if anchored:
            for col, val in default_values.items():
                traj.df[col] = val
        traj.df["anchored"] = anchored
    return trajectories
