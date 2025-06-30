import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np


if __name__ == "__main__":
    path = "C:/Users/Ben/shipwise/data_base/AIS/kiel/3_features/traj.parquet"
    df_traj = pd.read_parquet(path, engine="pyarrow")
    df_traj = df_traj[df_traj["mmsi"] == 209339000]
    df_traj = df_traj[["mmsi", "timestamp", "lon", "lat", "speed", "course", "outlier"]]

    print(df_traj.head())

    plt.figure(figsize=(8, 8))
    plt.scatter(df_traj["lon"], df_traj["lat"], s=1, alpha=0.5, color="blue")
    df_out = df_traj[df_traj["outlier"]]
    plt.scatter(df_out["lon"], df_out["lat"], s=1, alpha=1, color="red")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("All AIS Positions")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
