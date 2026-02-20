import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path("data/gmm")
K = 10
for k in range(K):
    tif_path = folder / f"density_cluster_{k}.tif"

    with rasterio.open(tif_path) as src:
        grid = src.read(1)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.clip(0, 100), origin="lower")
    plt.title(f"GMM Cluster {k} Density")
    plt.colorbar(label="Log Density (standardized)")
    plt.tight_layout()
    plt.savefig(f"GMM_Cluster_{k}.png")