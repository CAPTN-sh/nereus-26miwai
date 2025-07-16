from movingpandas import Trajectory
from typing import Iterable
from shapely.ops import nearest_points
from vincenty import vincenty
from utils.map_reader import get_map


def filter_by_map(trajectories: Iterable[Trajectory], map_path, map_name):
    geofence = get_map(map_path, map_name)
    for traj in trajectories:
        traj.df = traj.df[traj.df["geometry"].within(geofence)]
    return trajectories


def add_within(trajectories: Iterable[Trajectory], map_path, map_name, col_name):
    geofence = get_map(map_path, map_name)
    for traj in trajectories:
        traj.df[col_name] = (
            traj.df["geometry"].apply(lambda x: x.within(geofence)).astype(int)
        )
    return trajectories


def add_min_distance(trajectories: Iterable[Trajectory], map_path, map_name, col_name):
    geofence = get_map(map_path, map_name)
    for traj in trajectories:
        traj.df["nearest_point"] = traj.df["geometry"].apply(
            lambda x: nearest_points(geofence, x)[0]
        )
        traj.df[col_name] = traj.df.apply(
            lambda x: vincenty(
                (x.geometry.y, x.geometry.x),
                (x.nearest_point.y, x.nearest_point.x),
                miles=False,
            ),
            axis=1,
        )
        traj.df.drop(["nearest_point"], axis=1, inplace=True)
    return trajectories
