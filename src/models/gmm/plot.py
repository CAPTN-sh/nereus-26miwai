from pathlib import Path
import torch

import numpy as np
import torch
from models.utils.maps.scene_gernerator import SceneLoader
from models.utils.maps.rasterize import Rasterizer

from utils.config import DATA_FOLDER_PATH
import matplotlib.pyplot as plt
from pathlib import Path


path = DATA_FOLDER_PATH / "maps/2_standardized/fh_10/kiel/"
sl = SceneLoader(Rasterizer([10.12, 54.31, 10.33, 54.46]))

path = Path("/home/bbi/nereus/nereus/data/gmm")

density_contiguous = np.ascontiguousarray(sl.load_cluster(path, 16))
density_maps = torch.from_numpy(density_contiguous).to(torch.float32).squeeze(0)

for channel in range(density_maps.shape[0]):

    map_c = density_maps[channel].cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(map_c, origin="lower", cmap="turbo")
    plt.title(f"Density Channel {channel}")
    plt.colorbar(label="Log Density (standardized)")
    plt.tight_layout()
    plt.savefig(f"density_channel_{channel}.png")
    plt.close()