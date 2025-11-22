# plot_heatmap.py
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(file_name: str, pred: np.ndarray, target: np.ndarray = None):
    os.makedirs("images", exist_ok=True)

    X, Y = pred.shape  # (X, Y)

    fig, ax = plt.subplots(figsize=(int(X / 10), int(Y / 10)))

    im = ax.imshow(
        pred.T,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap="viridis",
        alpha=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability", rotation=90)

    if target is not None:
        mask = (target > 0).astype(float)

        # contour expects the same orientation as imshow content, so use mask.T
        ax.contour(
            mask.T,
            levels=[0.5],
            colors="red",
            linewidths=0.8,
            origin="lower",
        )

    # Grid cosmetics
    ax.set_xlim(-0.5, X - 0.5)
    ax.set_ylim(-0.5, Y - 0.5)
    ax.set_xlabel("Grid X (column index)")
    ax.set_ylabel("Grid Y (row index)")
    ax.set_title(f"Heat Map {file_name}")

    out_path = os.path.join("images", f"{file_name}.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
