from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data.graph.build_dataloader import graph_loader
from models.traisformer.model import TrAISformer
from utils.config import AIS_FOLDER_PATH, MAP_FOLDER_PATH, TRAIN_BBOX

from data.map.rasterize import Rasterizer
from data.map.scene_gernerator import SceneLoader
from models.gmm.model import AIS_GMM
from models.gmm.utils_grid import cluter_to_grid

def load_training_data(device):
    train_loader, _ = graph_loader(
        data_folder=AIS_FOLDER_PATH,
        flag="train",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        batch_size= 512,
        pin_memory=True,
        pred_len= 1,
        obs_len= 60,
        max_edge_dist = 0,
    )

    sl = SceneLoader(Rasterizer(TRAIN_BBOX))
    scene_contiguous = np.ascontiguousarray(sl.load_scene(MAP_FOLDER_PATH))
    scene = torch.from_numpy(scene_contiguous).to(device).to(torch.float32)

    return train_loader, scene


def fit_gmm(train_loader, scene, device, n_clusters, traisformer_path):
    # load traisfromer
    ckpt = torch.load(traisformer_path, map_location=device)
    prior_module = TrAISformer(ckpt["config"]).to(device)
    prior_module.load_state_dict(ckpt["model_state_dict"])
    prior_module.eval()
    prior_module.requires_grad_(False)

    # fit gmm
    gmm = AIS_GMM(prior_module, n_clusters = n_clusters)
    gmm.fit(train_loader, scene=scene, device=device, max_samples = 10000)

    # save model
    state_dict_full = {
        "prior_config": gmm.prior_model.config,
        "prior_state_dict": gmm.prior_model.state_dict(),
        "gmm": gmm.gmm,
        "n_clusters": gmm.k
    }
    torch.save(state_dict_full, f"checkpoints/gmm/cluster_{n_clusters}/ais_gmm.pt")

    return gmm

if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    train_loader, scene = load_training_data(device)

    n_clusters = 16
    traisformer_path = Path("checkpoints/traisformer/traisformer_dest_best.pt")
    gmm = fit_gmm(train_loader, scene, device, n_clusters, traisformer_path)
    grids = cluter_to_grid(gmm, train_loader, scene, device, n_clusters)


"""
CUDA_VISIBLE_DEVICES=1 python src/models/gmm/train.py
"""