import os
import numpy as np

import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from utils.config import METER_CRS

def plot_mdn_k_trajectories(
    file_name,
    obs_pos,        # [T_obs, 2]
    fut_pos,        # [T_fut, 2]
    pred_abs_k,     # [K, T_fut, 2]
    pi_k,
    region,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    # ---- background map ----
    map_path = f"/home/bbi/nereus/assets/maps/2_standardized/dma_10/{region}/land.geojson"
    background = gpd.read_file(map_path).to_crs(METER_CRS)
    background.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

    # ---- observed ----
    ax.scatter(
        obs_pos[:, 0],
        obs_pos[:, 1],
        color="blue",
        marker="o",
        s=7,
        alpha=0.7,
        label="Observed",
    )

    # ---- ground truth future ----
    ax.scatter(
        fut_pos[:, 0],
        fut_pos[:, 1],
        color="green",
        marker="o",
        s=7,
        alpha=0.7,
        label="Ground Truth",
    )

    # ---- K predicted trajectories ----
    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    K = pred_abs_k.shape[0]
    for k in range(K):
        traj = pred_abs_k[k]
        color = cmap(norm(pi_k[k]))

        ax.scatter(
            traj[:, 0],
            traj[:, 1],
            color=color,
            s=7,
            alpha=0.7,
            zorder=2,
            label=f"Mode {k} (π={pi_k[k]:.2f})"
        )

    # ---- compute bounds from trajectories ----
    all_x = np.concatenate([
        obs_pos[:, 0],
        fut_pos[:, 0],
        pred_abs_k[:, :, 0].reshape(-1)
    ])

    all_y = np.concatenate([
        obs_pos[:, 1],
        fut_pos[:, 1],
        pred_abs_k[:, :, 1].reshape(-1)
    ])

    minx, maxx = all_x.min(), all_x.max()
    miny, maxy = all_y.min(), all_y.max()

    # Add small padding (meters)
    pad = 200  # adjust (100–500 works well for AIS)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    # Keep correct aspect ratio (IMPORTANT for map coordinates)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"MDN K-Trajectory Prediction in {region}")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Mode Probability (π)")

    ax.legend()
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{file_name}.png", dpi=200)
    plt.close()