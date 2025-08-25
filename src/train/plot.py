from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_traj(
    file_name,
    obs_pos,
    fut_pos,
    pred_pos_rel,
    seq_start_end,
):
    
    fig, ax = plt.subplots(figsize=(10, 10))

    map_path = Path("data/kiel/maps/kiel_districts.geojson")
    background = gpd.read_file(map_path).to_crs("EPSG:32632")
    background.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

    start_abs = obs_pos[:, :, -1].unsqueeze(2)
    y_pred_abs = start_abs + pred_pos_rel.cumsum(dim=2)

    def plot_trajectories(ax, traj, i, color):
        traj_i = traj[i].detach().permute(1, 0).cpu()
        xs = traj_i[:, 0].numpy()
        ys = traj_i[:, 1].numpy()
        ax.scatter(xs, ys, color=color, alpha=0.4, s=2)

    for i, (s, e) in enumerate(seq_start_end):
        if i > 5:
            break
        plot_trajectories(ax, obs_pos, s, color="blue")
        plot_trajectories(ax, fut_pos, s, color="green")
        plot_trajectories(ax, y_pred_abs, s, color="red")

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
    ax.set_title("AIS Positions (EPSG:32632)")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.savefig(f"prediction_{file_name}.png")
    plt.close()
