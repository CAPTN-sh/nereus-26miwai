import numpy as np
import torch
import torch.nn as nn
from typing import Dict


class TorchCoordsNormalizer(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def normalize_coords(self, coords):
        return (coords - self.mean) / self.std

    def denormalize_coords(self, coords_norm):
        return coords_norm * self.std + self.mean


class Normalizer:
    def __init__(self):
        self.stats: Dict[str, Dict[str, np.ndarray]] = {}

    def approximate_from_df(self, df, columns):
        for column in columns:
            self.stats[column] = {
                "mean": np.array(df[column].mean()),
                "std": np.array(df[column].std()),
                "min": np.array(df[column].min()),
                "max": np.array(df[column].max()),
            }

    def load_from_file(self, path):
        self.stats = np.load(path, allow_pickle=True).item()

    def save_to_file(self, path):
        np.save(path, self.stats)

    def normalize(self, values, key):
        return (values - self.stats[key]["mean"]) / self.stats[key]["std"]

    def denormalize(self, values, key):
        return (values * self.stats[key]["std"]) + self.stats[key]["mean"]

    def to_TorchCoordsNormalizer(self):
        mean = [self.stats["lat"]["mean"], self.stats["lon"]["mean"]]
        std = [self.stats["lat"]["std"], self.stats["lon"]["std"]]
        return TorchCoordsNormalizer(mean, std)
