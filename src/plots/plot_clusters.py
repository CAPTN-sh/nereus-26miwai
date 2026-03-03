import matplotlib.pyplot as plt

def plot_clusters(grids):
    n_clusters = grids.shape[0]

    for k in range(n_clusters):
        grid = grids[k]

        plt.figure(figsize=(6, 8))
        plt.imshow(
            grid,
            origin="lower",
            cmap="turbo"
        )
        plt.colorbar(label="Normalized Density")
        plt.title(f"GMM Cluster {k}")
        plt.xlabel("Grid Cell X")
        plt.ylabel("Grid Cell Y")
        plt.tight_layout()
        plt.savefig(f"checkpoints/gmm/cluster_{n_clusters}/density_cluster_{k}.png")