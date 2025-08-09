from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from desire.nn.loss import *
from desire.utils.normalizer import TorchCoordsNormalizer


def plot_traj(
    file_name,
    obs_traj,
    pred_traj_gt,
    y_pred_traj,
    seq_start_end,
    normalizer: TorchCoordsNormalizer,
):
    fig, ax = plt.subplots(figsize=(10, 10))

    map_path = Path("desire-pytorch/kiel_districts.geojson")
    background = gpd.read_file(map_path).to_crs("EPSG:4326")
    background.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

    start_abs = obs_traj[:, :, -1].unsqueeze(2)
    y_pred_abs = start_abs + y_pred_traj.cumsum(dim=2)

    def plot_trajectories(ax, traj, i, color):
        traj_i = traj[i].detach().permute(1, 0)
        traj_i = normalizer.denormalize_coords(traj_i).cpu()
        xs = traj_i[:, 1].numpy()
        ys = traj_i[:, 0].numpy()
        ax.scatter(xs, ys, color=color, alpha=0.4, s=2)

    for i, (s, e) in enumerate(seq_start_end):
        if i > 5:
            break
        plot_trajectories(ax, obs_traj, s, color="blue")
        plot_trajectories(ax, pred_traj_gt, s, color="green")
        plot_trajectories(ax, y_pred_abs, s, color="red")

    legend_elements = [
        Line2D([0], [0], color="blue", label="Observed"),
        Line2D([0], [0], color="green", label="Ground Truth"),
        Line2D([0], [0], color="red", label="Predicted"),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlim(10.125, 10.3)
    ax.set_ylim(54.32, 54.45)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("AIS Positions")
    plt.grid(True)
    plt.savefig(f"prediction_{file_name}.png")
    plt.close()
