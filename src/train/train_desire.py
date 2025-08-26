import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from lazy_loader.loader import lazy_loader
from models.desire.desire import DESIRE
from models.desire.nn.loss import k_total_loss
from models.desire.utils.params import DESIREParams
from train.eval import eval

os.environ['OMP_NUM_THREADS'] = '4'

def train_worker(
    rank: int,
    world_size: int,
    local_rank: int,
    data_folder: Path,
    scene_path: Path,
    scene_meta_path: Path,
    batch_size: int = 2048,
    num_epochs: int = 10,
    norm_clip_value: float = 1.0,
    lr: float = 4e-3,
):
    ### --- DDP Setup --- ###
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}] Starting training on {device}")

    ### --- DataLoader --- ###
    train_dset, train_sampler, train_loader = lazy_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2022-04-14"),
        max_date=pd.Timestamp("2022-04-16"),
        world_size=world_size,
        rank=rank,
        batch_size=batch_size,
        pin_memory=True
    )

    eval_dset, eval_sampler, eval_loader = lazy_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2022-06-01"),
        max_date=pd.Timestamp("2022-06-01"),
        world_size=world_size,
        rank=rank,
        batch_size=batch_size,
        pin_memory=True
    )

    if rank == 0:
        print("additional features:", train_dset.feature_cols)
        print("There are {} traj loaded for training".format(len(train_dset)))
        print("There are {} traj loaded for evaluation".format(len(eval_dset)))

    ### --- Model --- ###

    npz = np.load(scene_path)
    scene = torch.from_numpy(npz["I"]).unsqueeze(0).to(device) #TODO unsqueeze?
    scene_meta = json.load(open(scene_meta_path))

    params = DESIREParams()
    model = DESIRE(params).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 4, gamma=0.5)

    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        sum_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad(set_to_none=True)

            obs_feat, obs_pos, obs_pos_rel, fut_pos, fut_pos_rel, seq_start_end = [
                tensor.to(device) for tensor in batch
            ]
            obs_pos_last = obs_pos[:, :, -1].to(device)
            with amp.autocast(device_type="cuda"):
                pred_pos_rel, pred_pos_rel_refined, mean, log_var, scores = model(
                    obs_feat,
                    obs_pos_last,
                    obs_pos_rel,
                    fut_pos_rel,
                    seq_start_end,
                    scene,
                    scene_meta,
                )

                tloss, (l_reg, l_kld, l_ioc, l_ref) = k_total_loss(
                    pred_pos_rel, pred_pos_rel_refined, fut_pos_rel, mean, log_var, scores
                )
                final_loss = tloss

            scaler.scale(final_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), norm_clip_value)
            scaler.step(optimizer)
            scaler.update()

            sum_loss += final_loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = sum_loss / max(1, num_batches)

        if rank == 0:
            print(
                f"[Epoch {epoch}] total_loss={avg_loss:.6f} "
                f"l_reg={l_reg.item():.6f}  l_kld={l_kld.item():.6f}  l_ioc={l_ioc.item():.6f}   l_ref={l_ref.item():.6f}"
            )

        eval(f"eval_plot_{epoch}", model.module, eval_loader, device, scene, scene_meta)

    dist.destroy_process_group()


def get_distributed_args():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return rank, world_size, local_rank


if __name__ == "__main__":
    rank, world_size, local_rank = get_distributed_args()

    data_folder = Path("/home/bbiesenbach/data/kiel/ais/3_features")
    scene_path = Path("data/kiel/scenes/bev.npz")
    scene_meta_path = Path("data/kiel/scenes/bev_meta.json")

    train_worker(
        rank, world_size, local_rank,
        data_folder=data_folder,
        scene_path=scene_path,
        scene_meta_path=scene_meta_path,
        batch_size = 128,
        num_epochs = 30
    )