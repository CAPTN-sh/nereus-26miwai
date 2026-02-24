import torch.nn as nn
from torch_geometric.data import Data

class DensityIntent(nn.Module):
    def __init__(self, density_maps):
        super().__init__()
        self.density_maps = density_maps

    def forward(self, data: Data, map = None):
        ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
        #["sailing", "cargo", "passenger", "other"]
        ship_group = data.static[ego_idx, -4:]
        ship_idx = ship_group.argmax(dim=-1)
        return self.density_maps[ship_idx].unsqueeze(1), None