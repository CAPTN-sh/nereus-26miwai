import torch
from torch import nn
from torch_geometric.data import Data

from models.gmm.model import AIS_GMM


class DensityMap(nn.Module):
    """Map module that selects precomputed density map depending on ship group.
    """

    def __init__(self, density_maps):
        super().__init__()
        self.density_maps = density_maps

    def forward(self, data: Data, scene = None):
        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
        #["sailing", "cargo", "passenger", "other"]
        ship_group = data.static[ego_idx, -4:]
        ship_idx = ship_group.argmax(dim=-1)
        return self.density_maps[ship_idx].unsqueeze(1), None

class MAP_GMM(nn.Module):
    """Map module that selects precomputed density map depending on cluster id.
    The cluster id is determined through a gmm on a pretrained trAISformer.
    """

    def __init__(self, gmm: AIS_GMM, cluster_maps):
        super().__init__()
        self.gmm = gmm
        self.cluster_maps = cluster_maps

    def forward(self, data, scene):
        cluster_prob = self.gmm.predict_proba(data, scene)
        cluster_prob = torch.tensor(cluster_prob, device=self.cluster_maps.device)

        density_map = torch.einsum(
            "bk,khw->bhw",
            cluster_prob,
            self.cluster_maps
        ).unsqueeze(1)

        return density_map, None
