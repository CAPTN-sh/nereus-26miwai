import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt
import numpy as np
from models.utils.maps.rasterize import Rasterizer
import matplotlib.pyplot as plt

from utils.config import DATA_FOLDER_PATH

MAX_DIST = 2000
MAX_DEPTH = 20

def process_maps(rasterizer: Rasterizer, base_path):
    # 1. Setup
    target_shape = (rasterizer.y_size, rasterizer.x_size)
    affine = rasterio.transform.from_origin(rasterizer.x_min, rasterizer.y_max, rasterizer.pos_res, rasterizer.pos_res)
    # TODO from config
    crs_target = "EPSG:25832"
    
    # Hilfsfunktion für GeoJSON (Vektor -> Distanz-Grid)
    def geojson_to_dist(file_name):
        gdf = gpd.read_file(base_path / file_name)
        # In das Ziel-System (UTM) transformieren
        gdf = gdf.to_crs(crs_target)
        
        # Rasterize
        mask = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=target_shape,
            transform=affine,
            fill=0,
            all_touched=True
        )
        # Euklidische Distanz (in Pixeln, dann mal Auflösung = Meter)
        dist_m = distance_transform_edt(mask == 0) * rasterizer.pos_res
        dist_m = np.minimum(dist_m, MAX_DIST)
        dist_log = np.log1p(dist_m) / np.log1p(MAX_DIST)

        return dist_log

    # Hilfsfunktion für TIFF (Raster -> Resampled Grid)
    def tiff_to_grid(file_name):
        with rasterio.open(base_path / file_name) as src:
            out_data = np.zeros(target_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=out_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=affine,
                dst_crs=crs_target,
                resampling=Resampling.bilinear,
                dst_nodata=-9999
            )
            invalid_mask = (out_data == -9999)
            out_data = 1.0 - out_data.astype(np.float32).clip(0, MAX_DEPTH) / MAX_DEPTH
            ind = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
            return out_data[tuple(ind)]

    # Layer verarbeiten
    print("Processing Scene Layers...")
    land_dist = geojson_to_dist("land.geojson")
    restricted_dist = geojson_to_dist("restricted_area.geojson")
    ferry_dist = geojson_to_dist("ferry_route.geojson")
    depth = tiff_to_grid("water_depth.tif")

    # Stacken (C, H, W)
    # Wichtig: y-Achse oft flippen, falls TIFF top-down aber Grid bottom-up ist
    map_stack = np.stack([land_dist, restricted_dist, ferry_dist, depth])
    map_stack = np.flip(map_stack, axis=1)
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
path = DATA_FOLDER_PATH / "maps/2_standardized/fh/kiel/"
my_rasterizer = Rasterizer([10.12, 54.31, 10.33, 54.46])

final_maps = process_maps(my_rasterizer, path)
plot_maps(final_maps, my_rasterizer)

print(f"Grid Size: {my_rasterizer.x_size}x{my_rasterizer.y_size}")
print(f"Map Stack Shape: {final_maps.shape}")