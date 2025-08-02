import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as cx
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from utils.map_loader import MapLoader
from utils.config import Config

DEFAULT_CRS = "EPSG:4326"
PLOT_CRS = "EPSG:3857"
CALCULATE_CRS = "EPSG:32632"

if __name__ == "__main__":
    Config("src/preprocessing/configs/_main.yaml")
    maps = MapLoader()

    paths = [
        "C:/Users/Ben/shipwise/data/kiel/AIS/2_decoded/20221025_traj.parquet",
        "C:/Users/Ben/shipwise/data/kiel/AIS/2_decoded/20220501_traj.parquet",
    ]
    df_traj = pd.concat([pd.read_parquet(path, engine="pyarrow") for path in paths])
    df_traj = gpd.GeoDataFrame(
        df_traj,
        geometry=gpd.points_from_xy(df_traj["lon"], df_traj["lat"]),
        crs=DEFAULT_CRS,
    )
    df_traj = df_traj[df_traj["geometry"].within(maps.get_layer("water").geometry)]
    df_traj = df_traj.to_crs(PLOT_CRS)

    path = "C:/Users/Ben/shipwise/data/kiel/AIS/3_features/nodes.parquet"
    df_nodes = gpd.read_parquet(path).to_crs(PLOT_CRS)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {
        "water": "lightblue",
        "waterways": "yellow",
        "restricted": "red",
        "marina": "green",
        "districts": "brown",
        "ferry_stop": "white",
    }

    maps = MapLoader()
    for name, layer in maps.get_layers().geometry.items():
        geo_web = gpd.GeoSeries(layer, crs=DEFAULT_CRS).to_crs(PLOT_CRS)
        geo_web.plot(
            ax=ax, color=colors[name], alpha=0.5, edgecolor="black", linewidth=0.5
        )

    df_nodes = df_nodes[df_nodes["ship_type"] != 36].copy()

    # target_mmsis = [211865680]
    target_mmsis = pd.Series(df_nodes["mmsi"].unique()).sample(10, random_state=42)
    df_nodes = df_nodes[df_nodes["mmsi"].isin(target_mmsis)].copy()

    for mmsi in target_mmsis:
        df = df_nodes[df_nodes["mmsi"] == mmsi]
        ax.scatter(df.geometry.x, df.geometry.y, s=1, alpha=0.5, color="blue")
        df = df_traj[df_traj["mmsi"] == mmsi]
        ax.scatter(df.geometry.x, df.geometry.y, s=1, alpha=0.5, color="red")

    first3 = df_nodes.groupby("traj_id").head(3)
    last3 = df_nodes.groupby("traj_id").tail(3)

    gpd.GeoSeries(first3["geometry"], crs=PLOT_CRS).plot(
        ax=ax, markersize=3, alpha=0.5, color="green", label="First 3"
    )
    gpd.GeoSeries(last3["geometry"], crs=PLOT_CRS).plot(
        ax=ax, markersize=3, alpha=0.5, color="orange", label="Last 3"
    )

    ax.set_xlim(maps.total_bounds(PLOT_CRS)[[0, 2]])
    ax.set_ylim(maps.total_bounds(PLOT_CRS)[[1, 3]])
    # cx.providers.Esri.WorldStreetMap
    # cx.providers.OpenStreetMap.Mapnik
    # cx.providers.Esri.WorldImagery
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("AIS Positions")
    plt.grid(True)
    plt.savefig(f"data/kiel/maps/map_layers.png")
    plt.show()
