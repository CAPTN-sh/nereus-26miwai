import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter
from models.traisformer.hierarchical_loss import RASTER

DPI = 150
PX_PER_CELL = 3

def plot_heatmap(file_name: str, 
                 density: np.ndarray, 
                 obs_pos: np.ndarray = None,
                 fin_pos: np.ndarray = None,
                 epoch = 0):
    os.makedirs("images", exist_ok=True)
    X, Y = density.shape

    # Heatmap area (pixel-perfect)
    ax_w_in = (X * PX_PER_CELL) / DPI
    ax_h_in = (Y * PX_PER_CELL) / DPI

    # Fixed margins for labels/title (in inches)
    left_in   = 0.80
    bottom_in = 0.75
    top_in    = 0.55
    right_in  = 0.80

    # Colorbar size/padding (in inches)
    cbar_w_in = 0.2
    cbar_pad_in = 0.12

    fig_w_in = left_in + ax_w_in + cbar_pad_in + cbar_w_in + right_in
    fig_h_in = bottom_in + ax_h_in + top_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)

    # Place the heatmap axes so its WIDTH/HEIGHT are exact
    ax_left = left_in / fig_w_in
    ax_bottom = bottom_in / fig_h_in
    ax_width = ax_w_in / fig_w_in
    ax_height = ax_h_in / fig_h_in
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    im = ax.imshow(
        density.T * 1000,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap="viridis",
        zorder=1  # Heatmap ganz unten
    )

    if obs_pos is not None:
        x, y = RASTER.pos_to_grid_coords(obs_pos.detach().cpu().numpy())
        ax.scatter(x, y, color="black", s=1)

    if fin_pos is not None:
        x, y = RASTER.pos_to_grid_coords(fin_pos.detach().cpu().numpy())
        ax.scatter(x, y, color="red", s=1)

    ax.set_xlim(-0.5, X - 0.5)
    ax.set_ylim(-0.5, Y - 0.5)

    ax.set_xlabel("Grid X (lon)")
    ax.set_ylabel("Grid Y (lat)")
    ax.set_title(f"Heat Map {file_name} e{epoch}")

    # Colorbar axes to the right (does not shrink heatmap)
    cax_left  = (left_in + ax_w_in + cbar_pad_in) / fig_w_in
    cax_width = cbar_w_in / fig_w_in
    cax = fig.add_axes([cax_left, ax_bottom, cax_width, ax_height])

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Probability", rotation=90, labelpad=10)
    cbar.ax.tick_params(pad=6)

    # --- Force colorbar tick labels to show in e-3 (×10^-3) ---
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, -3))  # always 10^-3
    cbar.ax.yaxis.set_major_formatter(fmt)
    cbar.update_ticks()

    out_path = os.path.join("images", f"{file_name}.png")
    fig.savefig(out_path)  # IMPORTANT: no bbox_inches="tight"
    plt.close(fig)
