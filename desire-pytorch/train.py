import os
import logging
import sys
import time
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from desire.data.loader import data_loader
from desire.utils.misc import relative_to_abs, get_dset_path
from desire.utils.misc import int_tuple, bool_flag, get_total_norm
from desire.models import DESIRE
from desire.utils.params import IOCParams, SGMParams
from desire.nn.loss import *
from PIL import Image
import subprocess
import re


from PIL import Image
import torchvision.transforms.functional as TF
from torch import amp

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def get_freer_gpu():
    try:
        result = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
        # Find all memory usage lines using regex
        matches = re.findall(r'(\d+)MiB / +(\d+)MiB', result)
        free_memory = [int(total) - int(used) for used, total in matches]

        if not free_memory:
            raise ValueError("Could not parse GPU memory from nvidia-smi")

        return int(np.argmax(free_memory))

    except Exception as e:
        print(f"[WARN] Could not detect GPU memory: {e}")
        return None


def train(dataset_name,
          path_of_static_image,
          restore_path=None,
          batch_size=64,
          num_epochs=700,
          norm_clip_value=1.0,
          lr = 5e-4):

    train_path = get_dset_path(dataset_name, 'train')

    logger.info("Initializing train dataset")
    logger.info(train_path)
    train_dset, train_loader = data_loader(train_path, batch_size=batch_size, delim = ",")
    gpu_id = get_freer_gpu()
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu")
    logger.info("Device is %s", device)

    iterations_per_epoch = len(train_dset) // batch_size
    if num_epochs:
        num_iterations = int(iterations_per_epoch * num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    desire = DESIRE(IOCParams(),
                    SGMParams())
    desire = desire.to(device)

    image = Image.open(path_of_static_image)
    scene = TF.to_tensor(image)
    scene.unsqueeze_(0)
    scene = scene.to(device)

    optimizer = optim.Adam(desire.parameters(),lr=lr)

    # Maybe restore from checkpoint
    if restore_path is not None:
        restore_dict = torch.load(restore_path)
        desire.load_state_dict(restore_dict)

    scene = scene.to(device)
    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            logging.info("epoch {} :batch_idx {}, ".format(epoch, batch_idx))
            optimizer.zero_grad()
            
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, _, seq_start_end) = batch

            obs_traj = obs_traj.permute(1,2,0)
            pred_traj_gt = pred_traj_gt.permute(1,2,0)

            obs_traj_rel = obs_traj_rel.permute(1,2,0)
            pred_traj_gt_rel = pred_traj_gt_rel.permute(1,2,0)

            x_start = obs_traj[:, :, 0].to(device)
            with amp.autocast(device_type='cuda'):
                y_pred_traj, pred_delta, mean, log_var = desire(obs_traj_rel,
                                                                pred_traj_gt_rel,
                                                                x_start,
                                                                scene,
                                                                seq_start_end)

                tloss, (l2l,kld, cel,rl) = total_loss(y_pred_traj,
                                                    pred_delta,
                                                    pred_traj_gt_rel,
                                                    mean,
                                                    log_var)
            scaler.scale(tloss.sum()).backward()
            torch.nn.utils.clip_grad_norm_(desire.parameters(), norm_clip_value)
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 10 == 0:
                logging.info("Total loss {}; epoch = {}; batch_idx = {}".format(
                    str(tloss.item()), epoch, batch_idx))
                logging.info("L2L {}; RL {}; CEL {}; KLD {};".format(
                    l2l.item(), rl.item(), cel.item(), kld.item()))
        weight_save_path = "weights/iter_{}.pth".format(str(epoch).zfill(3))
        logging.info("Saving weights for epoch {} in {}".format(epoch, weight_save_path))
        torch.save(desire.state_dict(), weight_save_path)
        logging.info("Done saving weights for epoch {} in {}".format(epoch, weight_save_path))

if __name__ == "__main__":
    print(os.getcwd())
    mp.set_start_method('spawn', force=True)
    dataset_name = os.path.abspath("./dataset/denmark/")
    path_of_static_image = os.path.abspath("./bg.png")
    train(dataset_name, path_of_static_image, batch_size=256, num_epochs=40, lr = 1e-4)
