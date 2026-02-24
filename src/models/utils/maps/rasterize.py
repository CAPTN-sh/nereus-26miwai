import numpy as np
import pyproj
import torch
import torch.nn as nn

class Rasterizer(nn.Module):
    def __init__(self, bbox, pos_res = 50):
        super().__init__()
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:25832"), always_xy=True
        )

        self.pos_res = pos_res
        self.x_min, self.y_min = np.floor(transformer.transform(*bbox[:2]))
        self.x_max, self.y_max = np.ceil(transformer.transform(*bbox[2:]))
        self.x_size = int((self.x_max - self.x_min) / self.pos_res)
        self.y_size = int((self.y_max - self.y_min) / self.pos_res)

        self.sog_res = 1
        self.sog_max = 40
        self.sog_size = int(self.sog_max / self.sog_res)

        self.cog_res = 5
        self.cog_max = 360
        self.cog_size = int(self.cog_max / self.cog_res)

        # bins generated from quantiles (training set "kiel")
        self.register_buffer("acc_bins", torch.tensor([-0.070, -0.034, -0.017, -0.009, -0.005, 0.005, 0.009, 0.017, 0.034, 0.070]))
        self.register_buffer("rot_bins", torch.tensor([-2.9, -1.6, -1.0, -0.7, -0.5, 0.5, 0.7, 1.0, 1.6, 2.9]))

        self.acc_size = len(self.acc_bins) + 1
        self.rot_size = len(self.rot_bins) + 1

    def get_total_grid_sizes(self):
        return self.x_size, self.y_size, self.sog_size, self.cog_size, self.acc_size, self.rot_size

    def pos_to_index(self, pos):
        x, y = self.pos_to_grid_coords(pos)

        x_idx = x.floor().clamp(0, self.x_size - 1).to(torch.long)
        y_idx = y.floor().clamp(0, self.y_size - 1).to(torch.long)

        return x_idx, y_idx
    
    def pos_to_grid_coords(self, pos):
        x = (pos[..., 0] - self.x_min) / self.pos_res
        y = (pos[..., 1] - self.y_min) / self.pos_res

        return x, y

    def kin_to_index(self, feat):
        sog_idx = (feat[..., 0] / self.sog_res).floor()
        sog_idx = sog_idx.clamp(0, self.sog_size - 1).to(torch.long)

        cog_idx = (feat[..., 1] / self.cog_res).floor()
        cog_idx = cog_idx.clamp(0, self.cog_size - 1).to(torch.long)

        return sog_idx, cog_idx

    def dyn_to_index(self, feat):
        acc_idx = torch.bucketize(feat[..., 0].contiguous(), self.acc_bins)
        rot_idx = torch.bucketize(feat[..., 1].contiguous(), self.rot_bins)
        return acc_idx, rot_idx