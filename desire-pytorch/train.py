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
        num_workers=10,
        min_date="2022-04-15",
        max_date="2022-04-15",  # 227002
        # max_date="2022-06-07",
    )
    print("additional features:", train_dset.add_feats)

    eval_dset, eval_loader = data_loader(
        nodes_path,
        edges_path,
        normalizer=normalizer,
        batch_size=batch_size,
        num_workers=10,
        min_date="2022-06-08",
        max_date="2022-06-08",
        # max_date="2022-06-15",
    )

    gpu_id = get_freer_gpu()
    device = torch.device(
        f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu"
    )
    logger.info("Device is %s", device)

    iterations_per_epoch = len(train_dset) // batch_size

    logger.info("There are {} traj loaded".format(len(train_dset)))
    logger.info("There are {} iterations per epoch".format(iterations_per_epoch))

    normalizer = normalizer.to_TorchNormalizer().to(device)

    sgm_params = SGMParams()
    sgm_params.rnn_enc_x_params.input_size = 2 + len(train_dset.add_feats)

    desire = DESIRE(IOCParams(), sgm_params, normalizer)
    desire = desire.to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        desire = torch.nn.DataParallel(desire)

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
    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        sum_loss = 0
        num_batches_total = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # logging.info("epoch {} :batch_idx {}, ".format(epoch, batch_idx))
            optimizer.zero_grad()

            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end = [
                tensor.to(device) for tensor in batch
            ]

            obs_traj = obs_traj.permute(1, 2, 0)
            pred_traj_gt = pred_traj_gt.permute(1, 2, 0)

            obs_traj_rel = obs_traj_rel.permute(1, 2, 0)
            pred_traj_gt_rel = pred_traj_gt_rel.permute(1, 2, 0)

            x_start = obs_traj[:, :, 0].to(device)
            with amp.autocast(device_type="cuda"):
                y_pred_traj, pred_delta, mean, log_var = desire(
                    obs_traj_rel, pred_traj_gt_rel, x_start, scene, seq_start_end
                )

            tloss, (l2l, kld, cel, rl) = total_loss(
                y_pred_traj, pred_delta, pred_traj_gt_rel, mean, log_var
            )
            num_batches = seq_start_end.size(0)
            final_loss = torch.zeros(num_batches)
            for i, (s, e) in enumerate(seq_start_end):
                s = s.item()
                # e = e.item()
                # l = tloss[s:e].sum()
                l = tloss[s]
                final_loss[i] = l
            final_loss = final_loss.sum()

            """
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(desire.parameters(), norm_clip_value)
            optimizer.step()
            """

            scaler.scale(final_loss).backward()
            torch.nn.utils.clip_grad_norm_(desire.parameters(), norm_clip_value)
            scaler.step(optimizer)
            scaler.update()

            sum_loss += final_loss.item() / num_batches
            num_batches_total += 1

        loss_str = str(sum_loss / num_batches_total)

        logging.info("Total loss {}; epoch = {}".format(loss_str, epoch))
        logging.info(
            "L2L {}; RL {}; CEL {}; KLD {};".format(
                l2l.item(), rl.item(), cel.item(), kld.item()
            )
        )

        if epoch % 1 != 0:
            continue

        evaluate(epoch, desire, eval_loader, device, scene, normalizer)

        if False:
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


def evaluate(epoch, desire, eval_loader, device, scene, normalizer):
    desire.eval()
    total_loss_val = 0
    total_l2l, total_kld, total_cel, total_rl = 0, 0, 0, 0
    num_batches_total = 0

    for batch in tqdm(eval_loader, desc="Evaluating"):
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end = [
            tensor.to(device) for tensor in batch
        ]

        obs_traj = obs_traj.permute(1, 2, 0)
        pred_traj_gt = pred_traj_gt.permute(1, 2, 0)
        obs_traj_rel = obs_traj_rel.permute(1, 2, 0)
        pred_traj_gt_rel = pred_traj_gt_rel.permute(1, 2, 0)

        x_start = obs_traj[:, :, 0]
        with amp.autocast(device_type="cuda"):
            y_pred_traj, pred_delta, mean, log_var = desire(
                obs_traj_rel, pred_traj_gt_rel, x_start, scene, seq_start_end
            )

        tloss, (l2l, kld, cel, rl) = total_loss(
            y_pred_traj, pred_delta, pred_traj_gt_rel, mean, log_var
        )
        num_batches = seq_start_end.size(0)
        final_loss = torch.zeros(num_batches)
        for i, (s, e) in enumerate(seq_start_end):
            s = s.item()
            final_loss[i] = tloss[s]
        total_loss_val += final_loss.sum().item() / num_batches
        num_batches_total += 1

    avg_loss = total_loss_val / num_batches_total

    logger.info(f"[Eval] Avg Loss: {avg_loss:.4f}")

    plot_traj(
        epoch, avg_loss, obs_traj, pred_traj_gt, y_pred_traj, seq_start_end, normalizer
    )

    desire.train()  # restore training mode afterward
    return avg_loss


def plot_traj(
    epoch,
    loss,
    obs_traj,
    pred_traj_gt,
    y_pred_traj,
    seq_start_end,
    normalizer: TorchNormalizer,
):
    fig, ax = plt.subplots(figsize=(10, 10))

    map_path = Path("desire-pytorch/kiel_districts.geojson")
    background = gpd.read_file(map_path).to_crs("EPSG:4326")
    background.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

    start_abs = obs_traj[:, :, -1].unsqueeze(2)
    y_pred_abs = start_abs + y_pred_traj.cumsum(dim=2)

    def plot_trajectories(ax, traj, i, color):
        traj_i = traj[i].detach().permute(1, 0)
        traj_i = normalizer.denormalize(traj_i).cpu()
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
    plt.savefig(f"prediction_{epoch}_{loss}.png")
    plt.close()


if __name__ == "__main__":
    print(os.getcwd())
    mp.set_start_method("spawn", force=True)
    nodes_path = Path(
        "/home/bbiesenbach/data/kiel/ais/3_features/nodes.parquet"
    ).resolve()
    edges_path = Path(
        "/home/bbiesenbach/data/kiel/ais/3_features/edges.parquet"
    ).resolve()
    path_of_static_image = Path("scene_encoded.png").resolve()
    coord_norm_path = Path("normalization_stats.npy").resolve()

    image = Image.open(path_of_static_image)

    train(
        nodes_path,
        edges_path,
        path_of_static_image,
        coord_norm_path,
        batch_size=8192,
        num_epochs=20,
        lr=5e-3,
    )
