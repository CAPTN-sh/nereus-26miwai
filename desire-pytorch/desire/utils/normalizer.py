import numpy as np
import torch
import torch.nn as nn


class TorchNormalizer(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def normalize(self, coords):
        return (coords - self.mean) / self.std

    def denormalize(self, coords_norm):
        return coords_norm * self.std + self.mean


class CoordsNormalizer:
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def approximate(self, df):
        self.mean = np.array(df[["lat", "lon"]].mean())
        self.std = np.array(df[["lat", "lon"]].std())

    def load_from_file(self, path):
        norm_stats = np.load(path, allow_pickle=True).item()
        self.mean = norm_stats["mean"]
        self.std = norm_stats["std"]

    def save_to_file(self, path):
        norm_stats = {"mean": self.mean, "std": self.std}
        np.save(path, norm_stats)

    def normalize(self, latlon):
        return (latlon - self.mean) / self.std

    def denormalize(self, latlon_norm):
        return (latlon_norm * self.std) + self.mean

    def to_TorchNormalizer(self):
        return TorchNormalizer(self.mean, self.std)
