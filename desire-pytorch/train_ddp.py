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

from desire.lazy_loader.trajectories import LazyTrajectoryDataset, seq_collate
from desire.models import DESIRE
from desire.utils.params import IOCParams, SGMParams
from desire.utils.normalizer import CoordsNormalizer, TorchNormalizer
from desire.nn.loss import *
from PIL import Image
import subprocess
import re
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import argparse


from PIL import Image
import torchvision.transforms.functional as TF
from torch import amp

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def train_worker(
    rank,
    world_size,
    local_rank,
    batch_size=8192,
    num_epochs=30,
    norm_clip_value=1.0,
    lr=5e-3,
):
    ### --- Step 1: DDP Setup --- ###
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}] Starting training on {device}")

    ### --- Step 2: DataLoader --- ###

    nodes_path = Path("/home/bbiesenbach/data/kiel/ais/3_features/nodes.parquet")
    edges_path = Path("/home/bbiesenbach/data/kiel/ais/3_features/edges.parquet")
    path_of_static_image = Path("scene_encoded.png").resolve()

    normalizer = CoordsNormalizer()
    normalizer.load_from_file(Path("normalization_stats.npy").resolve())

    train_dset = LazyTrajectoryDataset(
        nodes_path,
        edges_path,
        normalizer,
        min_date="2022-04-15",
        max_date="2022-04-15",
    )

    train_sampler = DistributedSampler(
        train_dset, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_loader = DataLoader(
        train_dset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=10,
        collate_fn=seq_collate,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        print("additional features:", train_dset.feature_cols)
        print("There are {} traj loaded".format(len(train_dset)))

    ### --- Step 3: Model --- ###

    sgm_params = SGMParams()
    sgm_params.rnn_enc_x_params.input_size = 2 + len(train_dset.feature_cols)
    normalizer = normalizer.to_TorchNormalizer().to(device)

    desire = DESIRE(IOCParams(), sgm_params, normalizer).to(device)
    desire = DDP(desire, device_ids=[rank], find_unused_parameters=True)

    image = Image.open(path_of_static_image)
    scene = TF.to_tensor(image)
    scene.unsqueeze_(0)
    scene = scene.to(device)

    optimizer = optim.Adam(desire.parameters(), lr=lr)
    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        sum_loss = 0
        num_batches_total = 0
        train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
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
            final_loss = torch.zeros(num_batches, device=device)
            for i, (s, e) in enumerate(seq_start_end):
                s = s.item()
                # e = e.item()
                # l = tloss[s:e].sum()
                l = tloss[s]
                final_loss[i] = l
            final_loss = final_loss.sum()

            scaler.scale(final_loss).backward()
            torch.nn.utils.clip_grad_norm_(desire.parameters(), norm_clip_value)
            scaler.step(optimizer)
            scaler.update()

            sum_loss += final_loss.item() / num_batches
            num_batches_total += 1

        loss_str = str(sum_loss / num_batches_total)

        if rank == 0:
            print("Total loss {}; epoch = {}".format(loss_str, epoch))
            print(
                "L2L {}; RL {}; CEL {}; KLD {};".format(
                    l2l.item(), rl.item(), cel.item(), kld.item()
                )
            )

    dist.destroy_process_group()
    # TODO Ensure evaluation, plotting, and checkpointing only happen on rank 0


def get_distributed_args():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return rank, world_size, local_rank


if __name__ == "__main__":
    rank, world_size, local_rank = get_distributed_args()
    train_worker(rank, world_size, local_rank)
