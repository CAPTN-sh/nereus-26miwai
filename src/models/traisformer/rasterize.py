import numpy as np
import pyproj
import torch

from .params import TraisformerParams


class Rasterizer:

    def __init__(self, config: TraisformerParams):
        self.pos_res = 50
        self.sog_res = 1
        self.cog_res = 5

        # TODO transformer from config
        bbox = [10.12, 54.31, 10.33, 54.46]
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:25832"), always_xy=True
        )

        self.x_min, self.y_min = np.floor(transformer.transform(*bbox[:2]))
        self.x_max, self.y_max = np.ceil(transformer.transform(*bbox[2:]))
        self.sog_max = 40
        self.cog_max = 360

        self.x_size = int((self.x_max - self.x_min) / self.pos_res)
        self.y_size = int((self.y_max - self.y_min) / self.pos_res)
        self.sog_size = int(self.sog_max / self.sog_res)
        self.cog_size = int(self.cog_max / self.cog_res)

    def get_total_gird_sizes(self):
        return self.x_size, self.y_size, self.sog_size, self.cog_size

    def pos_to_index(self, pos):
        x_idx = ((pos[:, 0, :] - self.x_min) / self.pos_res).floor()
        x_idx = x_idx.clamp(0, self.x_size - 1).to(torch.long)

        y_idx = ((pos[:, 1, :] - self.y_min) / self.pos_res).floor()
        y_idx = y_idx.clamp(0, self.y_size - 1).to(torch.long)

        return x_idx, y_idx

    def feat_to_index(self, feat):
        sog_idx = (feat[:, 0, :] / self.sog_res).floor()
        sog_idx = sog_idx.clamp(0, self.sog_size - 1).to(torch.long)

        cog_idx = (feat[:, 1, :] / self.cog_res).floor()
        cog_idx = cog_idx.clamp(0, self.cog_size - 1).to(torch.long)

        return sog_idx, cog_idx
