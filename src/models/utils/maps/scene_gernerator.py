import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt
import numpy as np
from models.utils.maps.rasterize import Rasterizer
import matplotlib.pyplot as plt
from pathlib import Path
from utils.config import DATA_FOLDER_PATH

# TODO from config
MAX_DIST = 2000
MAX_DEPTH = 20
CRS_TARGET = "EPSG:25832"

class SceneLoader():
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
        with rasterio.open(path) as src:
            out_data = np.zeros(self.target_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=out_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.affine,
                dst_crs=CRS_TARGET,
                resampling=Resampling.bilinear,
                dst_nodata=-9999
            )
            invalid_mask = (out_data == -9999)
            out_data = 1.0 - out_data.astype(np.float32).clip(0, MAX_DEPTH) / MAX_DEPTH
            ind = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
            return out_data[tuple(ind)]
    
    def geojson_to_dist(self, path):
        if path is None or not Path(path).exists():
            return np.ones(self.target_shape, dtype=np.float32)
        gdf = gpd.read_file(path)
        # In das Ziel-System (UTM) transformieren
        gdf = gdf.to_crs(CRS_TARGET)
        
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
        # Layer verarbeiten
        print("Processing Scene Layers...")
        land_dist = self.geojson_to_dist(base_path / "land.geojson")
        restricted_dist = self.geojson_to_dist(base_path / "restricted_area.geojson")
        ferry_dist = self.geojson_to_dist(base_path / "ferry_route.geojson")
        depth = self.tiff_to_grid(base_path / "water_depth.tif")

        map_stack = np.stack([land_dist, restricted_dist, ferry_dist, depth])
        map_stack = np.flip(map_stack, axis=1)
        return map_stack
    
    def load_density(self, base_path: Path):
        # Layer verarbeiten
        print("Processing Density Layers...")
        sailing = self.tiff_to_grid(base_path / "density_sailing.tif")
        cargo = self.tiff_to_grid(base_path / "density_cargo.tif")
        passenger = self.tiff_to_grid(base_path / "density_passenger.tif")
        other = self.tiff_to_grid(base_path / "density_other.tif")

        map_stack = np.stack([sailing, cargo, passenger, other])
        map_stack = np.flip(map_stack, axis=1)
        return map_stack
    
    def load_cluster(self, base_path: Path, n_cluster):
        folder = base_path / f"cluster_{n_cluster}"
        map_stack = np.stack([
            self.tiff_to_grid(folder / f"density_cluster_{i}.tif") 
            for i in range(n_cluster)
        ])
        return map_stack

def plot_maps(map_stack, rasterizer):
    layer_names = ["Land Dist", "Restricted Dist", "Ferry Dist", "Water Depth"]
    num_layers = map_stack.shape[0]
    
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))

    for i in range(num_layers):
        # Plotte das Raster (origin='lower' passt zum np.flip)
        im = axes[i].imshow(map_stack[i], origin='lower', cmap='viridis', extent=[0, rasterizer.x_size, 0, rasterizer.y_size])
        axes[i].set_title(layer_names[i])
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("map_validation_with_polygons.png")

# --- Ausführung ---
if __name__ == "__main__":
    path = DATA_FOLDER_PATH / "maps/2_standardized/fh_10/kiel/"
    my_rasterizer = Rasterizer([10.12, 54.31, 10.33, 54.46])

    final_maps = process_maps(my_rasterizer, path)
    plot_maps(final_maps, my_rasterizer)

    print(f"Grid Size: {my_rasterizer.x_size}x{my_rasterizer.y_size}")
    print(f"Map Stack Shape: {final_maps.shape}")