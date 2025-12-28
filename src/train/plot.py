from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_traj(
    file_name,
    obs_pos,
    fut_pos,
    pred_pos,
    seq_start_end,
):

    fig, ax = plt.subplots(figsize=(50, 50))

    # TODO config
    map_path = Path("data/maps/2_standardized/fhkiel_train/kiel/land.geojson")
    background = gpd.read_file(map_path).to_crs("EPSG:25832")
    background.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

    def plot_trajectories(ax, traj, i, color):
        traj_i = traj[i].detach().permute(1, 0).cpu()
        xs = traj_i[:, 0].numpy()
        ys = traj_i[:, 1].numpy()
        ax.scatter(xs, ys, color=color, alpha=0.7, s=2)

    for i, (s, e) in enumerate(seq_start_end):
        if i > 5:
            break
        plot_trajectories(ax, obs_pos, i, color="blue")
        plot_trajectories(ax, fut_pos, i, color="green")
        plot_trajectories(ax, pred_pos, i, color="red")

    legend_elements = [
        Line2D([0], [0], color="blue", label="Observed"),
        Line2D([0], [0], color="green", label="Ground Truth"),
        Line2D([0], [0], color="red", label="Predicted"),
    ]
    ax.legend(handles=legend_elements)

    minx, miny, maxx, maxy = background.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("AIS Positions (EPSG:25832)")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.savefig(f"images/{file_name}.png")
    plt.close()
