from sklearn.cluster import KMeans

import pandas as pd
import torch
from torch import nn
from loaders.graph_loader.loader import graph_loader
from pathlib import Path
from utils.config import DATA_FOLDER_PATH
from models.dae.model import DAE
import os
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

class DAE_KMeans(nn.Module):
    def __init__(self, dae: DAE, k):
        super().__init__()
        self.dae = dae
        self.kmeans = KMeans(n_clusters=k, n_init=20)

    def fit(self, train_loader, device, max_samples):
        self.dae.eval()

        all_z = []
        total_samples = 0

        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)

                z = self.dae.inference(batch)
                z = torch.nn.functional.normalize(z, dim=1)
                all_z.append(z.cpu())
                total_samples += z.size(0)

                if total_samples >= max_samples:
                    break

        z_cpu = torch.cat(all_z, dim=0).numpy()
        self.kmeans.fit(z_cpu)

    def predict(self, data):
        self.dae.eval()

        with torch.no_grad():
            z = self.dae.inference(data)
            z = torch.nn.functional.normalize(z, dim=1)
            z = z.cpu().numpy()

        cluster_id = self.kmeans.predict(z)
        return cluster_id

if __name__ == "__main__":

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    train_loader, _ = graph_loader(
        data_folder=DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel",
        flag="train",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        batch_size= 512,
        pin_memory=True,
        pred_len= 1,
        obs_len= 30,
        max_edge_dist = 0,
    )

    print("load dae")
    best_ckpt_path = Path("checkpoints/traisformer") / f"DAE_best.pt"
    ckpt = torch.load(best_ckpt_path, map_location=device)
    dae = DAE(ckpt["config"])
    dae.load_state_dict(ckpt["model_state_dict"])

    print("fit k_means")
    k_means = DAE_KMeans(dae)
    k_means.fit(train_loader, device=device, max_samples = 1000)

    print("fit k_means")
    with torch.no_grad():
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            cluster_id = k_means.predict(batch)
    