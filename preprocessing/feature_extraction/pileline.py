from pathlib import Path
import yaml
import importlib
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


class FeatureExtractionPipeline:

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def run(self):
        df = self.load_decoded_data()

        groups = [group for mmsi, group in df.groupby("mmsi")]
        with ProcessPoolExecutor() as executor:
            dfs = list(
                tqdm(executor.map(self._process_ship, groups), total=len(groups))
            )
        df = pd.concat(dfs, ignore_index=True)

        df.to_parquet(self.config["out_file"], index=True, engine="pyarrow")

    def load_decoded_data(self):
        files = Path(self.config["in_folder"]).glob("*_traj.parquet")
        dfs = [pd.read_parquet(file, engine="pyarrow") for file in files]
        df = pd.concat(dfs, ignore_index=True)
        return df

    def _process_ship(self, df):
        for step in self.config["pipeline"]:
            func = self._import_from_string(step["function"])
            df = func(**{**step.get("args", {}), **{"df": df}})
        return df

    def _import_from_string(self, dotted_path):
        module_path, func_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
