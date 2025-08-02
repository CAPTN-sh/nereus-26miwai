import os
import logging
import sys
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from desire.data_loader.loader import data_loader
from desire.models import DESIRE
from desire.utils.params import IOCParams, SGMParams
from desire.utils.normalizer import CoordsNormalizer, TorchNormalizer
from desire.nn.loss import *
from PIL import Image
import subprocess
import re
from tqdm import tqdm


from PIL import Image
import torchvision.transforms.functional as TF
from torch import amp

FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_freer_gpu():
    try:
        result = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
        # Find all memory usage lines using regex
        matches = re.findall(r"(\d+)MiB / +(\d+)MiB", result)
        free_memory = [int(total) - int(used) for used, total in matches]

        if not free_memory:
            raise ValueError("Could not parse GPU memory from nvidia-smi")

        return int(np.argmax(free_memory))

    except Exception as e:
        print(f"[WARN] Could not detect GPU memory: {e}")
        return None


def train(
    nodes_path,
    edges_path,
    path_of_static_image,
    coord_norm_path,
    restore_path=None,
    batch_size=64,
    num_epochs=700,
    norm_clip_value=1.0,
    lr=5e-4,
):

    logger.info("Initializing train dataset")

    normalizer = CoordsNormalizer()
    normalizer.load_from_file(coord_norm_path)

    train_dset, train_loader = data_loader(
        nodes_path,
        edges_path,
        normalizer=normalizer,
        batch_size=batch_size,
        loader_num_workers=4,
    )
    gpu_id = get_freer_gpu()
    device = torch.device(
        f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu"
    )
    logger.info("Device is %s", device)

    iterations_per_epoch = len(train_dset) // batch_size
    if num_epochs:
        num_iterations = int(iterations_per_epoch * num_epochs)

    logger.info("There are {} traj loaded".format(len(train_dset)))
    logger.info("There are {} iterations per epoch".format(iterations_per_epoch))

    normalizer = normalizer.to_TorchNormalizer().to(device)

    sgm_params = SGMParams()
    sgm_params.rnn_enc_x_params.input_size = 2 + 8

    desire = DESIRE(IOCParams(), sgm_params, normalizer)
    desire = desire.to(device)

    image = Image.open(path_of_static_image)
    scene = TF.to_tensor(image)
    scene.unsqueeze_(0)
    scene = scene.to(device)

    optimizer = optim.Adam(desire.parameters(), lr=lr)

    # Maybe restore from checkpoint
    if restore_path is not None:
        restore_dict = torch.load(restore_path)
        desire.load_state_dict(restore_dict)

    scene = scene.to(device)
    # scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # logging.info("epoch {} :batch_idx {}, ".format(epoch, batch_idx))
            optimizer.zero_grad()

            # TODO right order?
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end = [
                tensor.to(device) for tensor in batch
            ]

            obs_traj = obs_traj.permute(1, 2, 0)
            pred_traj_gt = pred_traj_gt.permute(1, 2, 0)

            obs_traj_rel = obs_traj_rel.permute(1, 2, 0)
            pred_traj_gt_rel = pred_traj_gt_rel.permute(1, 2, 0)

            x_start = obs_traj[:, :, 0].to(device)
            # with amp.autocast(device_type="cuda"):
            y_pred_traj, pred_delta, mean, log_var = desire(
                obs_traj_rel, pred_traj_gt_rel, x_start, scene, seq_start_end
            )

            tloss, (l2l, kld, cel, rl) = total_loss(
                y_pred_traj, pred_delta, pred_traj_gt_rel, mean, log_var
            )
            num_batches = seq_start_end.size(0)
            final_loss = torch.zeros(num_batches)
            for i, (s, e) in enumerate(seq_start_end[0:-2]):
                s = s.item()
                e = e.item()
                l = tloss[s:e].sum()
                final_loss[i] = l
            final_loss = final_loss.sum()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(desire.parameters(), norm_clip_value)
            optimizer.step()

        loss_str = str(tloss.mean().item())

        logging.info(
            "Total loss {}; epoch = {}; batch_idx = {}".format(
                loss_str, epoch, batch_idx
            )
        )
        logging.info(
            "L2L {}; RL {}; CEL {}; KLD {};".format(
                l2l.item(), rl.item(), cel.item(), kld.item()
            )
        )

        if epoch % 10 != 0:
            continue

        plot_traj(epoch, loss_str, obs_traj, pred_traj_gt, y_pred_traj, normalizer)

        weight_save_path = "desire-pytorch/weights/iter_{}.pth".format(
            str(epoch).zfill(3)
        )
        logging.info(
            "Saving weights for epoch {} in {}".format(epoch, weight_save_path)
        )
        torch.save(desire.state_dict(), weight_save_path)
        logging.info(
            "Done saving weights for epoch {} in {}".format(epoch, weight_save_path)
        )


def plot_traj(
    epoch, loss, obs_traj, pred_traj_gt, y_pred_traj, normalizer: TorchNormalizer
):
    fig, ax = plt.subplots(figsize=(10, 10))

    map_path = Path("desire-pytorch/kiel_districts.geojson")
    background = gpd.read_file(map_path).to_crs("EPSG:4326")
    background.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

    start_abs = obs_traj[:, :, -1].unsqueeze(2)
    y_pred_abs = start_abs + y_pred_traj.cumsum(dim=2)

    def plot_trajectories(ax, traj, color):
        for i in range(min(3, traj.shape[0])):
            traj_i = traj[i].detach().permute(1, 0)
            traj_i = normalizer.denormalize(traj_i).cpu()
            xs = traj_i[:, 1].numpy()
            ys = traj_i[:, 0].numpy()
            ax.scatter(xs, ys, color=color, alpha=0.4, s=2)

    plot_trajectories(ax, obs_traj, color="blue")
    plot_trajectories(ax, pred_traj_gt, color="green")
    plot_trajectories(ax, y_pred_abs, color="red")

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
    plt.savefig(f"prediction_{epoch}_{loss}.png")
    plt.close()


if __name__ == "__main__":
    print(os.getcwd())
    mp.set_start_method("spawn", force=True)

    nodes_path = Path("data/kiel/ais/3_features/nodes.parquet").resolve()
    edges_path = Path("data/kiel/ais/3_features/edges.parquet").resolve()
    path_of_static_image = Path("scene_encoded.png").resolve()
    coord_norm_path = Path("normalization_stats.npy").resolve()

    image = Image.open(path_of_static_image)

    train(
        nodes_path,
        edges_path,
        path_of_static_image,
        coord_norm_path,
        batch_size=4,
        num_epochs=100,
        lr=1e-4,
    )
