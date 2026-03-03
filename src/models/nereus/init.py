from pathlib import Path
import torch
import numpy as np

from data.map.scene_gernerator import SceneLoader
from models.nereus.prior_modules.prior import DensityMap, MAP_GMM
from models.traisformer.model import TrAISformer
from models.gmm.model import AIS_GMM
from models.nereus.social_modules.social import GAT, SocialPooling
from models.nereus.model import NEREUS
from models.nereus.map_modules.map import MapAttention, ScenePoolingCNN
from models.nereus.params import NEREUSParams
from data.map.rasterize import Rasterizer

from utils.config import DATA_FOLDER_PATH, AIS_SOURCE, TRAIN_BBOX

def load_density_prior(rasterizer, device, region):
    """
    Load density maps and create module that selects the map dependent on ship group.
    """
    path = DATA_FOLDER_PATH / f"maps/2_standardized/{AIS_SOURCE}_10/{region}/"
    sl = SceneLoader(rasterizer)
    density_contiguous = np.ascontiguousarray(sl.load_density(path))
    density_maps = torch.from_numpy(density_contiguous).to(device).to(torch.float32)
    return DensityMap(density_maps)

def load_traisformer_prior(device):
    """
    Load pretrained traisformer prior that predicts the full future path of the vessel.
    """
    best_ckpt_path = Path("checkpoints/traisformer/traisformer_path_best.pt")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    prior_module = TrAISformer(ckpt["config"])
    prior_module.load_state_dict(ckpt["model_state_dict"])
    prior_module.eval()
    prior_module.requires_grad_(False)
    return prior_module

def load_gmm_prior(rasterizer, device, n_cluster = 16):
    """
    Load density maps created from gmm clustering and select dependent on cluster id.
    """
    best_ckpt_path = Path(f"checkpoints/gmm/cluster_{n_cluster}/ais_gmm.pt")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    trais = TrAISformer(ckpt["prior_config"]).to(device)
    trais.load_state_dict(ckpt["prior_state_dict"])
    trais.eval()
    trais.requires_grad_(False)

    ais_gmm = AIS_GMM(trais, n_clusters=ckpt["n_clusters"])
    ais_gmm.gmm = ckpt["gmm"]

    path = Path("checkpoints/gmm")
    sl = SceneLoader(rasterizer)

    cluster_contiguous = np.ascontiguousarray(sl.load_cluster(path, 16))
    cluster_maps = torch.from_numpy(cluster_contiguous).to(device).to(torch.float32)
    return MAP_GMM(ais_gmm, cluster_maps)

def init_nereus(model_name, cfg:NEREUSParams, device, rasterizer = None):
    """
    Build the nereus architecture dependent on the modules stated in the module_name.
    """
    if not rasterizer:
        rasterizer = Rasterizer(TRAIN_BBOX)

    prior_module = None
    if ("density" in model_name) or ("atte" in model_name):
        prior_module = load_density_prior(rasterizer, device, region="kiel")
    if ("path" in model_name):
        prior_module = load_traisformer_prior(device)
    if ("cluster" in model_name):
        prior_module = load_gmm_prior(rasterizer, device, n_cluster=16)

    social_module = None
    if ("gat" in model_name):
        social_module = GAT(cfg)
    if ("pool" in model_name):
        social_module = SocialPooling(cfg)

    map_module = None
    if ("cnn" in model_name):
        map_module = ScenePoolingCNN(rasterizer, cfg)
    if ("atte" in model_name):
        map_module = MapAttention(rasterizer, cfg)

    model = NEREUS(
        config = cfg,
        static_module = True,
        social_module = social_module,
        map_module = map_module,
        prior_module = prior_module,
    )
    return model