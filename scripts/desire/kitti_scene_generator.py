import json
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from affine import Affine
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

DIST_TO_COAST_CLIP_M   = 5000
RATE_CAP               = 1000
RESOLUTION             = 10     # meter per pixel
EXPOSURE_HOURS         = 100

def grid_from_params(params) -> Tuple[Affine, int, int, Tuple[float,float,float,float]]:
    xmin, ymin, xmax, ymax = (gpd.GeoSeries([box(*params["bounds"])], crs="EPSG:4326")
                              .to_crs(params["utm_crs"])
                              .total_bounds)
    W = int(np.ceil((xmax - xmin) / RESOLUTION))
    H = int(np.ceil((ymax - ymin) / RESOLUTION))
    # North-up, origin top-left: x = xmin + col*res, y = ymax - row*res
    transform = Affine.translation(xmin, ymax) * Affine.scale(RESOLUTION, -RESOLUTION)
    return transform, H, W, (xmin, ymin, xmax, ymax)

def rasterize_land(coast_utm: gpd.GeoDataFrame, transform: Affine, H: int, W: int) -> np.ndarray:
    shapes = [(geom, 1) for geom in coast_utm.geometry if geom.is_valid and not geom.is_empty]
    return rasterize(shapes=shapes, out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8)

def dist_to_coast_m(land_mask: np.ndarray) -> np.ndarray:
    water = (land_mask == 0).astype(np.uint8)
    dist_px = distance_transform_edt(water == 1)
    dist_m  = np.clip(dist_px * RESOLUTION, 0.0, DIST_TO_COAST_CLIP_M) / DIST_TO_COAST_CLIP_M
    return dist_m.astype(np.float32)

def rates_log1p(gdf: gpd.GeoDataFrame,
                bounds_utm: Tuple[float, float, float, float],
                H: int, W: int) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds_utm
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    cols = np.floor((xs - xmin) / RESOLUTION).astype(np.int64)
    rows = np.floor((ymax - ys) / RESOLUTION).astype(np.int64)

    counts = np.zeros((H, W), dtype=np.float32)
    np.add.at(counts, (rows, cols), 1.0)

    cell_area_km2 = (RESOLUTION * RESOLUTION) / 1e6
    rate = counts / (max(EXPOSURE_HOURS, 1e-6) * cell_area_km2)  # vessels / km^2 / hr
    rate = np.clip(rate, 0.0, RATE_CAP)
    rate = np.log1p(rate) / np.log1p(RATE_CAP)  # scaled to [0,1] (data-independent)
    return rate.astype(np.float32)

if __name__ == "__main__":
    # load params
    params_path = Path("configs/preprocessing/kiel_meta_data.yaml").resolve()
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # create grid
    transform, H, W, bounds_utm = grid_from_params(params)

    # load data
    coastline_path = Path("data/kiel/maps/kiel_districts.geojson").resolve()
    coast = gpd.read_file(coastline_path).to_crs("EPSG:4326").to_crs(params["utm_crs"])

    ais_path = Path("data/kiel/ais/3_features/nodes.parquet").resolve()
    df = pd.read_parquet(ais_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_pos(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(params["utm_crs"])

    # generate scene
    land  = rasterize_land(coast, transform, H, W).astype(np.float32)
    dist  = dist_to_coast_m(land)
    sail  = rates_log1p(gdf[gdf["sailing_vessel"]], bounds_utm, H, W)
    nsail = rates_log1p(gdf[~gdf["sailing_vessel"]], bounds_utm, H, W)

    scene = np.stack([land, dist, sail, nsail], axis=0).astype(np.float32)

    # save scene
    out_dir = Path("data/kiel/scenes").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "bev.npz", I=scene)

    # save meta data
    w2b = (~transform)
    world_to_bev = [[w2b.a, w2b.b, w2b.c],
                    [w2b.d, w2b.e, w2b.f],
                    [0.0,   0.0,   1.0]]
    meta = {
    "utm_crs": params["utm_crs"],
    "resolution": float(RESOLUTION),
    "size_px": [int(W), int(H)],
    "world_to_bev": world_to_bev,
    "feature_stride": 8,
    "channels": ["land_mask","dist_to_coast_m","sailing_rate","non_sailing_rate"]
    }
    with open(out_dir / "bev_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved: {out_dir}/bev.npz, {out_dir}/bev_meta.json")