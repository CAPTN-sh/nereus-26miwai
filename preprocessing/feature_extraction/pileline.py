import pandas as pd

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from preprocessing.utils.config import load_config
from preprocessing.utils.function_inport import import_from_string


class FeatureExtractionPipeline:

    def __init__(self, config_path):
        self.config = load_config(config_path)

    def run(self):
        """
        Run the preprocessing steps defined in the config.
        """
        df = self._load_decoded_data()
        for group_col in self.config["pipelines"]:
            df = self._run_parallel_pipeline(df, group_col)
        df.to_parquet(self.config["paths"]["out_file"], index=True, engine="pyarrow")

    def _load_decoded_data(self):
        files = Path(self.config["paths"]["in_folder"]).glob("*_traj.parquet")
        dfs = [pd.read_parquet(file, engine="pyarrow") for file in files]
        df = pd.concat(dfs, ignore_index=True)
        return df

    def _run_parallel_pipeline(self, df, group_col):
        groups = [group for _, group in df.groupby(group_col)]
        partial_fn = partial(self._run_pipeline, pipeline=group_col)
        with ProcessPoolExecutor() as executor:
            dfs = list(tqdm(executor.map(partial_fn, groups), total=len(groups)))
        df = pd.concat(dfs, ignore_index=True)
        return df

    def _run_pipeline(self, df, pipeline):
        steps = self.config["pipelines"][pipeline]
        for step in steps:
            func = import_from_string(step["function"])
            df = func(**{**step.get("args", {}), **{"df": df}})
        return df
