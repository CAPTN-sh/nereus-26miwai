import geopandas as gpd
from pathlib import Path
from utils.config import Config
from shapely import unary_union


def get_map(file, name):
    map_folder = Path(Config().folder["maps"]).resolve()

    map_file = gpd.read_file(map_folder / file)
    selected = map_file[map_file.name == name]
    geofence = unary_union(selected.geometry)

    return geofence
