import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import geopandas as gpd

from desire.data_loader.loader import data_loader
from desire.models import DESIRE
from desire.utils.params import IOCParams, SGMParams
from pathlib import Path

DEFAULT_CRS = "EPSG:4326"


def denormalize(traj, mean, std):
    # traj shape: [num_agents, 2, seq_len]
    traj_denorm = traj.clone()
    traj_denorm[:, 0, :] = traj[:, 0, :] * std[0] + mean[0]  # lat
    traj_denorm[:, 1, :] = traj[:, 1, :] * std[1] + mean[1]  # lon
    return traj_denorm


def main():
    # Config
    restore_model_path = Path("desire-pytorch/weights/iter_005.pth").resolve()
    path_of_static_image = Path("desire-pytorch/bg.png").resolve()
    nodes_path = Path("data/kiel/ais/3_features/nodes.parquet").resolve()
    edges_path = Path("data/kiel/ais/3_features/edges.parquet").resolve()

    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load scene image
    img = mpimg.imread(path_of_static_image)
    image = Image.open(path_of_static_image)
    width, height = image.size
    scene = TF.to_tensor(image).unsqueeze(0).to(device)

    # Load model
    desire = DESIRE(IOCParams(), SGMParams()).to(device)
    state_dict_checkpoint = torch.load(restore_model_path, map_location=device)
    desire.load_state_dict(state_dict_checkpoint)
    desire.eval()

    # Load one random batch (scene)
    train_dset, train_loader = data_loader(
        nodes_path, edges_path, batch_size=batch_size, loader_num_workers=4
    )

    print("samples", len(train_dset))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    for n in range(4):

        sample = next(iter(train_loader))
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            seq_start_end,
        ) = sample

        obs_traj = obs_traj.permute(1, 2, 0)
        pred_traj_gt = pred_traj_gt.permute(1, 2, 0)

        obs_traj_rel = obs_traj_rel.permute(1, 2, 0)
        pred_traj_gt_rel = pred_traj_gt_rel.permute(1, 2, 0)

        x_start = obs_traj[:, :, 0].to(device)

        # Inference
        with torch.no_grad():
            y_pred_traj, pred_deta = desire.inference(
                obs_traj_rel, scene, x_start, seq_start_end
            )

        start_abs = x_start.unsqueeze(2)
        y_pred_abs = start_abs + y_pred_traj.cumsum(dim=2)

        # Load stats
        norm_stats = np.load("normalization_stats.npy", allow_pickle=True).item()
        mean = norm_stats["mean"]
        std = norm_stats["std"]

        obs_traj_denorm = denormalize(obs_traj.cpu(), mean, std)
        pred_gt_denorm = denormalize(pred_traj_gt.cpu(), mean, std)
        pred_out_denorm = denormalize(y_pred_abs.cpu(), mean, std)

        map_path = Path("desire-pytorch/kiel_districts.geojson")
        background = gpd.read_file(map_path).to_crs(DEFAULT_CRS)
        background.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

        def plot_trajectories(ax, traj, color):
            for i in range(traj.shape[0]):  # iterate agents
                xs = traj[i, 1].numpy()
                ys = traj[i, 0].numpy()
                ax.scatter(xs, ys, color=color, alpha=0.5, s=5)

        plot_trajectories(ax, obs_traj_denorm, color="blue")
        plot_trajectories(ax, pred_gt_denorm, color="green")
        plot_trajectories(ax, pred_out_denorm, color="red")

        legend_elements = [
            Line2D([0], [0], color="blue", label="Observed"),
            Line2D([0], [0], color="green", label="Ground Truth"),
            Line2D([0], [0], color="red", label="Predicted"),
        ]
        ax.legend(handles=legend_elements)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("AIS Positions")
    plt.grid(True)
    plt.savefig("prediction.png")
    plt.show()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # optional but good on Windows
    main()
