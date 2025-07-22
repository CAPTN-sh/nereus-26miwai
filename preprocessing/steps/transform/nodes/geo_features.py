from movingpandas import Trajectory
from typing import Iterable
from utils.map_loader import MapLoader
from shapely.geometry import Point, Polygon
from typing import Tuple, List


def add_within_map(
    trajectories: Iterable[Trajectory], col_name: str, layer: str
) -> Iterable[Trajectory]:
    """
    Checks which points of the trajectorys are within the maps layers polygon.

    Args:
        trajectories (Iterable[Trajectory]): Traj to apply the function on.
        col_name (str): Name of the column storing boolean.
        layer (str): The maps layers name that store polygon.

    Return:
        Iterable[Trajectory]: Trajectories with additional column.
    """
    geofence = MapLoader().get_layer(layer).geometry
    for traj in trajectories:
        traj.df[col_name] = (
            traj.df["geometry"].apply(lambda x: x.within(geofence)).astype(int)
        )
    return trajectories


def add_distance_to_map_layer(
    trajectories: Iterable[Trajectory], col_name: str, layer: str
) -> Iterable[Trajectory]:
    """
    Finds the min distance of each trajectories points to a polygon it in meters.

    Args:
        trajectories (Iterable[Trajectory]): Traj to apply the function on.
        col_name (str): Name of the column storing the distance.
        layer (str): The maps layers name that store polygon.

    Return:
        Iterable[Trajectory]: Trajectories with additional column.
    """
    geofence = MapLoader().get_layer(layer, "EPSG:32632").geometry
    for traj in trajectories:
        df_geo = traj.df.to_crs("EPSG:32632").geometry
        traj.df[col_name] = df_geo.apply(lambda x: x.distance(geofence))
    return trajectories


def add_clostest_map_feature(
    trajectories: Iterable[Trajectory],
    col_name: str,
    at_index: int,
    layers: List[str],
    max_dist=float("inf"),
) -> Iterable[Trajectory]:
    """
    Finds the clostest polygon and the distance to it in meters.

    Args:
        trajectories (Iterable[Trajectory]): Traj to apply the function on.
        col_name (str): Name of the column storing the clostest polygons name.
        at_index (int): Index of the Point in the DataFrame.
        layers (List[str]): The maps layers names that store polygons.
        max_dist (float): Max distance at which the name "unknown" is stored.

    Return:
        str: Name of the clostest polygon.
        float: Distance to the clostest polygon in meters.
    """
    ploy_dict = MapLoader().get_features(layers, "EPSG:32632").geometry
    for traj in trajectories:
        df_geo = traj.df.to_crs("EPSG:32632").geometry
        closest_id, dist = _closest_polygon(df_geo.iloc[at_index], ploy_dict)
        traj.df[col_name] = closest_id if dist <= max_dist else "unknown"
    return trajectories


def _closest_polygon(point: Point, polygons: dict[str, Polygon]) -> Tuple[str, float]:
    """
    Finds the clostest polygon and the distance to it in meters.

    Args:
        point (Point): Point of interest.
        polygons (dict[str, Polygon]): Dict with the polygons and thier names.

    Return:
        str: name of the clostest polygon
        float: distance to the clostest polygon in meters
    """
    min_dist = float("inf")
    closest_id = None
    for poly_id, poly in polygons.items():
        distance = point.distance(poly)
        if distance < min_dist:
            min_dist = distance
            closest_id = poly_id

    return closest_id, min_dist
