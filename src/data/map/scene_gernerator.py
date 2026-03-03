import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt
import numpy as np
from data.map.rasterize import Rasterizer
from pathlib import Path

from utils.config import METER_CRS

MAX_DIST = 2000
MAX_DEPTH = 20

class SceneLoader():
    """Loades different scenes and stackes them into a np array."""

    def __init__(self, rasterizer: Rasterizer):
        self.rasterizer = rasterizer
        self.target_shape = (rasterizer.y_size, rasterizer.x_size)
        self.affine = rasterio.transform.from_origin(
            rasterizer.x_min, 
            rasterizer.y_max, 
            rasterizer.pos_res, 
            rasterizer.pos_res
        )

    def tiff_to_grid(self, path):
        """Loades a tif file, transforms it crs and bins it to the raster."""
        with rasterio.open(path) as src:
            out_data = np.zeros(self.target_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=out_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.affine,
                dst_crs=METER_CRS,
                resampling=Resampling.bilinear,
                dst_nodata=-9999
            )
            invalid_mask = (out_data == -9999)
            out_data = 1.0 - out_data.astype(np.float32).clip(0, MAX_DEPTH) / MAX_DEPTH
            ind = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
            return out_data[tuple(ind)]
    
    def geojson_to_dist(self, path):
        """Loades a geojson file, transforms it crs and bins it to the raster."""
        if path is None or not Path(path).exists():
            return np.ones(self.target_shape, dtype=np.float32)
        gdf = gpd.read_file(path)
        # In das Ziel-System (UTM) transformieren
        gdf = gdf.to_crs(METER_CRS)
        
        # Rasterize
        mask = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=self.target_shape,
            transform=self.affine,
            fill=0,
            all_touched=True
        )
        # Euklidische Distanz (in Pixeln, dann mal Auflösung = Meter)
        dist_m = distance_transform_edt(mask == 0) * self.rasterizer.pos_res
        dist_m = np.minimum(dist_m, MAX_DIST)
        dist_log = np.log1p(dist_m) / np.log1p(MAX_DIST)

        return dist_log

    def load_scene(self, base_path: Path):
        """Loades a the map context as a stacked np.array."""

        land_dist = self.geojson_to_dist(base_path / "land.geojson")
        restricted_dist = self.geojson_to_dist(base_path / "restricted_area.geojson")
        ferry_dist = self.geojson_to_dist(base_path / "ferry_route.geojson")
        depth = self.tiff_to_grid(base_path / "water_depth.tif")

        map_stack = np.stack([land_dist, restricted_dist, ferry_dist, depth])
        map_stack = np.flip(map_stack, axis=1)
        return map_stack
    
    def load_density(self, base_path: Path):
        """Loades a the density maps as a stacked np.array."""

        sailing = self.tiff_to_grid(base_path / "density_sailing.tif")
        cargo = self.tiff_to_grid(base_path / "density_cargo.tif")
        passenger = self.tiff_to_grid(base_path / "density_passenger.tif")
        other = self.tiff_to_grid(base_path / "density_other.tif")

        map_stack = np.stack([sailing, cargo, passenger, other])
        map_stack = np.flip(map_stack, axis=1)
        return map_stack
    
    def load_cluster(self, base_path: Path, n_cluster):
        """Loades a the density maps from the gmm clustering as a stacked np.array."""

        folder = base_path / f"cluster_{n_cluster}"
        map_stack = np.stack([
            self.tiff_to_grid(folder / f"density_cluster_{i}.tif") 
            for i in range(n_cluster)
        ])
        return map_stack