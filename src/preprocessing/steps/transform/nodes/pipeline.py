from pathlib import Path

import pandas as pd

from preprocessing.pipeline.pipeline import Pipeline
from preprocessing.steps.transform.nodes.trajectory import TrajectoryProcessor
from utils.config import Config


class NodesPipeline(Pipeline):
    def __init__(self):
        self.config = Config().get("transform_nodes")

    def load_tasks(self):

        paths = Path(self.config["paths"]["in_folder"]).glob("*_traj.parquet")
        dfs = [pd.read_parquet(path, engine="pyarrow") for path in paths]
        trajs_df = pd.concat(dfs).reset_index(drop=True)

        paths = Path(self.config["paths"]["in_folder"]).glob("ship_info*.parquet")
        dfs = [pd.read_parquet(path, engine="pyarrow") for path in paths]
        ships_df = pd.concat(dfs).reset_index(drop=True)

        # trajs_df = self._load_data("*_traj.parquet")
        # ships_df = self._load_data("ship_info*.parquet")

        task_count = trajs_df["mmsi"].nunique()

        def task_iter():
            for mmsi, traj_df in trajs_df.groupby("mmsi"):
                yield (mmsi, (traj_df, ships_df))

        return task_iter(), task_count

    def _load_data(self, glob_str):
        paths = Path(self.config["paths"]["in_folder"]).glob(glob_str)
        dfs = [pd.read_parquet(path, engine="pyarrow") for path in paths]
        df = pd.concat(dfs).reset_index(drop=True)
        return df

    def execut_task(self, mmsi, traj_df, ship_df):
        tp = TrajectoryProcessor(mmsi, traj_df, ship_df)

        for step in self.config["pipeline_steps"]:
            try:
                tp.run_pipeline_steps(step)
            except Exception as err:
                print(f"Error in step {step}: {err}")
                return
        try:
            return tp.get_df()
        except Exception as err:
            print(f"Error retrieving the trajectories: {err}")
            return

    def save_results(self, results: list[pd.DataFrame]):
        df = pd.concat(results, ignore_index=True)
        df["traj_id"] = df.groupby(["mmsi", "traj_id"], sort=False).ngroup()
        df.to_parquet(Path(self.config["paths"]["out_folder"]) / "nodes.parquet")
