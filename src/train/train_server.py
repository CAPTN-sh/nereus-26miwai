import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from lazy_loader.loader import lazy_loader
from models.desire.model import DESIRE
from models.desire.nn.loss import loss_desire
from models.desire.utils.params import DESIREParams
from models.lstm.model import LSTMModel
from train.eval import eval, eval_loss
from utils.logger import logger

os.environ["OMP_NUM_THREADS"] = "4"

def train_worker(
    dist_args,
    model,
    loss_fn,
    data_folder: Path,
    scene_path: Path,
    scene_meta_path: Path,
    batch_size: int = 128,
    num_epochs: int = 30,
    norm_clip_value: float = 1.0,
    lr: float = 4e-3,
    max_neighbors=10,
):
    ### --- DDP Setup --- ###
    rank, world_size, local_rank = dist_args
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    logging.info(f"[Rank {rank}] Starting training on {device}")

    ### --- DataLoader --- ###
    train_dset, train_sampler, train_loader = lazy_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2022-04-14"),
        max_date=pd.Timestamp("2022-04-16"),
        world_size=world_size,
        rank=rank,
        batch_size=batch_size,
        pin_memory=True,
        max_neighbors=max_neighbors,
    )

    eval_dset, eval_sampler, eval_loader = lazy_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2022-06-01"),
        max_date=pd.Timestamp("2022-06-01"),
        world_size=world_size,
        rank=rank,
        batch_size=batch_size,
        pin_memory=True,
        max_neighbors=max_neighbors,
    )

    if rank == 0:
        logging.info(f"[Train] model: {model.__class__.__name__}")
        # logging.info(f"additional features: {train_dset.feature_cols}")
        logging.info(f"There are {len(train_dset)} traj loaded for training")
        logging.info(f"There are {len(eval_dset)} traj loaded for evaluation")

    ### --- Model --- ###

    npz = np.load(scene_path)
    scene = torch.from_numpy(npz["I"]).unsqueeze(0).to(device)  # TODO unsqueeze?
    scene_meta = json.load(open(scene_meta_path))

    scene_meta["world_to_bev"] = torch.as_tensor(scene_meta["world_to_bev"], 
                                            device=device, 
                                            dtype=torch.float32)

    model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=num_epochs // 4, gamma=0.5
    )

    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        loss_sum_dict = {}
        loss_sum = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad(set_to_none=True)
            batch = [t.to(device) for t in batch]

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch, scene, scene_meta)
                loss, loss_dict = loss_fn(output, batch, epoch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), norm_clip_value)
            scaler.step(optimizer)
            scaler.update()

            num_batches += 1
            loss_sum += loss.item()
            for loss_name, loss_val in loss_dict.items():
                loss_sum_dict[loss_name] = (
                    loss_sum_dict.get(loss_name, 0.0) + loss_val.item()
                )

        scheduler.step()

        if rank == 0:
            loss_sum /= num_batches
            loss_print = f"[Epoch {epoch}] train_loss={loss_sum:.4f}"
            for loss_name, loss_val in loss_sum_dict.items():
                loss_val /= num_batches
                loss_print += f" {loss_name}={loss_val:.4f}"
            logging.info(loss_print)

        if (epoch + 1) % 4 == 0:
            eval(epoch, model.module, eval_loader, device, scene, scene_meta)
    dist.destroy_process_group()


def get_distributed_args():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return rank, world_size, local_rank


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_options = ["DESIRE", "LSTM"]
    model_choise = model_options[0]

    logger(file_prefix=f"train_server_{model_choise}")
    dist_args = get_distributed_args()

    if model_choise == "DESIRE":
        model = DESIRE(DESIREParams())
        loss_fn = loss_desire
        max_neighbors = 5

    if model_choise == "LSTM":
        model = LSTMModel(pred_len=36)
        loss_fn = eval_loss
        max_neighbors = 0

    data_folder = Path("/home/bbiesenbach/data/kiel/ais/3_features")
    scene_path = Path("data/kiel/scenes/bev.npz")
    scene_meta_path = Path("data/kiel/scenes/bev_meta.json")

    train_worker(
        dist_args,
        model=model,
        loss_fn=loss_fn,
        data_folder=data_folder,
        scene_path=scene_path,
        scene_meta_path=scene_meta_path,
        max_neighbors=max_neighbors,
        num_epochs = 64,
        batch_size = 4*2048,
    )