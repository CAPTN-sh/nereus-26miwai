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

from models.desire.model import DESIRE
from models.desire.nn.loss import loss_desire
from models.desire.utils.params import DESIREParams
from models.lstm.model import LSTMModel
from models.lstm.params import LSTMParams
from models.traisformer.loss import loss_intent_heatmap2
from models.traisformer.model import TrAISformer
from models.traisformer.params import TraisformerParams
from train.eval_heatmap import eval_heatmap
from scene_loader.loader import scene_loader
from train.eval import eval, eval_loss
from utils.logger import logger
from models.utils.maps.scene_gernerator import process_maps

from utils.config import DATA_FOLDER_PATH

os.environ["OMP_NUM_THREADS"] = "4"


def train_worker(
    dist_args,
    model,
    cfg,
    loss_fn,
    eval_fn,
    data_folder: Path,
    batch_size: int = 128,
    num_epochs: int = 30,
    norm_clip_value: float = 1.0,
    lr: float = 4e-3,
):
    ### --- DDP Setup --- ###
    rank, world_size, local_rank = dist_args
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    logging.info(f"[Rank {rank}] Starting training on {device}")

    ### --- DataLoader --- ###
    feat_cols = ["speed", "course", "acc", "angular_difference", "length", "width", "ship_group"]
    train_dset, train_sampler, train_loader = scene_loader(
        data_folder=data_folder,
        flag="val",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=True,
        feat_cols=feat_cols,
    )

    eval_dset, eval_loader = None, None

    ### --- Model --- ###
    model = model(cfg)

    if hasattr(model, "rasterizer"):
        print("loading scene layers")
        path = DATA_FOLDER_PATH / "maps/2_standardized/fh/kiel/" #TODO select scene depending on model
        scene_contiguous = np.ascontiguousarray(process_maps(model.rasterizer, path), dtype=np.float32)
        scene = torch.from_numpy(scene_contiguous).unsqueeze(0).to(device)
    else:
        scene = None

    model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank)

    if rank == 0:
        logging.info(f"[Train] model: {model.module.__class__.__name__}")
        # logging.info(f"additional features: {train_dset.feature_cols}")
        logging.info(f"There are {len(train_dset)} traj loaded for training")
        #logging.info(f"There are {len(eval_dset)} traj loaded for evaluation")

    # optimizer = optim.Adam(model.parameters(), lr=lr)
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
                output = model(batch, scene)
                loss, loss_dict = loss_fn(output, batch, epoch, None)

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

        # if (epoch + 1) % 4 == 0:
        eval_fn(epoch, model.module, eval_loader, device, scene)
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

    model_options = ["DESIRE", "LSTM", "TRAISFORMER"]
    model_choise = model_options[0]

    logger(file_prefix=f"train_server_{model_choise}")
    dist_args = get_distributed_args()

    if model_choise == "DESIRE":
        model = DESIRE
        cfg = DESIREParams()
        loss_fn = loss_desire
        eval_fn = eval

    if model_choise == "LSTM":
        model = LSTMModel
        cfg = LSTMParams()
        loss_fn = eval_loss
        eval_fn = eval

    if model_choise == "TRAISFORMER":
        model = TrAISformer
        cfg = TraisformerParams()
        loss_fn = loss_intent_heatmap2
        eval_fn = eval_heatmap

    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh/kiel"

    train_worker(
        dist_args,
        model=model,
        cfg=cfg,
        loss_fn=loss_fn,
        eval_fn=eval_fn,
        data_folder=data_folder,
        num_epochs = 10,
        batch_size = 64,
    )