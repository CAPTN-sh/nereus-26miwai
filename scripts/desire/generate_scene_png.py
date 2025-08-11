# %%
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import mapping

nodes_path = Path("data/kiel/ais/3_features/nodes.parquet").resolve()
map_path = Path("data/kiel/maps/kiel_districts.geojson").resolve()

# %%
nodes = pd.read_parquet(nodes_path)
sailing = nodes[nodes["sailing_vessel"]]
non_sailing = nodes[~nodes["sailing_vessel"]]

coast = gpd.read_file(map_path).to_crs("EPSG:4326")
# %%

width, height = 512, 512
xlim = (10.13, 10.32)
ylim = (54.31, 54.46)

scene_img = np.zeros((height, width, 3), dtype=np.uint8)


def lonlat_to_px(lon, lat):
    x_px = int((lon - xlim[0]) / (xlim[1] - xlim[0]) * width)
    y_px = int((lat - ylim[0]) / (ylim[1] - ylim[0]) * height)
    y_px = height - 1 - y_px  # invert y for image coordinate system
    return x_px, y_px


xres = (xlim[1] - xlim[0]) / width
yres = (ylim[1] - ylim[0]) / height
transform = Affine.translation(xlim[0], ylim[1]) * Affine.scale(xres, -yres)

shapes = [(mapping(geom), 1) for geom in coast.geometry if geom.is_valid]
mask = rasterize(
    shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.uint8,
)
scene_img[:, :, 0] = mask * 255


for lon, lat in zip(sailing["lon"], sailing["lat"]):
    x, y = lonlat_to_px(lon, lat)
    if 0 <= x < width and 0 <= y < height:
        scene_img[y, x, 1] = min(int(scene_img[y, x, 1]) + 1, 255)

for lon, lat in zip(non_sailing["lon"], non_sailing["lat"]):
    x, y = lonlat_to_px(lon, lat)
    if 0 <= x < width and 0 <= y < height:
        scene_img[y, x, 2] = min(int(scene_img[y, x, 2]) + 1, 255)

Image.fromarray(scene_img).save("data/kiel/scenes/scene_encoded.png")
