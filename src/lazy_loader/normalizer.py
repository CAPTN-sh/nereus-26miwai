from typing import Dict

import numpy as np


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
