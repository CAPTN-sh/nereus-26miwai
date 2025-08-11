import logging
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from models.desire.model import DESIRE
from src.lazy_loader.loader import lazy_loader
from src.models.desire.nn.loss import total_loss
from src.models.desire.utils.params import IOCParams, SGMParams
from src.train.plot import plot_traj


def train_worker(
    rank,
    world_size,
    local_rank,
    data_folder,
    scene_path,
    batch_size=2048,
    num_epochs=30,
    norm_clip_value=1.0,
    lr=1e-4,
):
    ### --- Step 1: DDP Setup --- ###
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}] Starting training on {device}")

    ### --- Step 2: DataLoader --- ###
    train_dset, train_sampler, train_loader = lazy_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2022-04-15"),
        max_date=pd.Timestamp("2022-04-20"),
        world_size=world_size,
        rank=rank,
        batch_size=batch_size,
    )

    eval_dset, eval_sampler, eval_loader = lazy_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2022-04-21"),
        max_date=pd.Timestamp("2022-04-21"),
        world_size=world_size,
        rank=rank,
        batch_size=batch_size,
    )

    if rank == 0:
        print("additional features:", train_dset.feature_cols)
        print("There are {} traj loaded for training".format(len(train_dset)))
        print("There are {} traj loaded for evaluation".format(len(eval_dset)))

    ### --- Step 3: Model --- ###

    sgm_params = SGMParams()
    sgm_params.rnn_enc_x_params.input_size = 2 + len(train_dset.feature_cols)
    normalizer = train_dset.normalizer.to_TorchCoordsNormalizer().to(device)

    desire = DESIRE(IOCParams(), sgm_params, normalizer).to(device)
    desire = DDP(desire, device_ids=[rank], find_unused_parameters=True)

    image = Image.open(scene_path)
    scene = TF.to_tensor(image)
    scene.unsqueeze_(0)
    scene = scene.to(device)

    optimizer = optim.Adam(desire.parameters(), lr=lr)
    #scaler = amp.GradScaler()

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
            # with amp.autocast(device_type="cuda"):
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
                e = e.item()
                final_loss[i] = tloss[s:e].sum()
            final_loss = final_loss.sum()

            final_loss.backward()
            nn.utils.clip_grad_norm_(desire.parameters(), norm_clip_value)
            optimizer.step()

            """
            scaler.scale(final_loss).backward()
            torch.nn.utils.clip_grad_norm_(desire.parameters(), norm_clip_value)
            scaler.step(optimizer)
            scaler.update()
            """

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

            if epoch % 1 == 0:
                eval(epoch, desire, eval_loader, device, scene, normalizer)

    dist.destroy_process_group()

def eval(epoch, desire, eval_loader, device, scene, normalizer):
    desire.eval()

    total_loss_val = 0
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
        #with amp.autocast(device_type="cuda"):
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

    print(f"[Eval] Avg Loss: {avg_loss:.4f}")

    plot_traj(epoch, obs_traj, pred_traj_gt, y_pred_traj, seq_start_end, normalizer)

    desire.train()
    return avg_loss


def get_distributed_args():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return rank, world_size, local_rank


if __name__ == "__main__":
    rank, world_size, local_rank = get_distributed_args()
    data_folder = Path("/home/bbiesenbach/data/kiel/ais/3_features")
    scene_path = Path("/home/bbiesenbach/data/kiel/scenes/scene_encoded.png")
    train_worker(rank, world_size, local_rank, data_folder, scene_path)
