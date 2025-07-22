from movingpandas import Trajectory
from typing import Iterable
from preprocessing.utils.pipeline.function_inport import import_from_string
import re
from datetime import timedelta


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
