from pathlib import Path
import pandas as pd
from multiprocessing import Manager, Lock
from concurrent.futures import ProcessPoolExecutor
from utils.config import Config
from preprocessing.steps.transform.nodes.trajectory import TrajectoryProcessor
from preprocessing.utils.ship_info_system.ship_info import ShipInfo
from preprocessing.utils.pipeline.pipeline import Pipeline

shared_ship_info = None


def init_worker(shared_db, lock):
    global shared_ship_info
    shared_ship_info = ShipInfo(shared_db, lock)


class NodesPipeline(Pipeline):
    def __init__(self):
        self.config = Config().get("transform_nodes")

    def init_pool(self, max_workers):
        return ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(Manager().dict(), Lock()),
        )

    def load_tasks(self):
        trajs_df = self._load_data("*_traj.parquet")
        ships_df = self._load_data("*_ship.parquet")

        ship_groups = dict(tuple(ships_df.groupby("mmsi")))
        task_count = trajs_df["mmsi"].nunique()

        def task_iter():
            for mmsi, traj_df in trajs_df.groupby("mmsi"):
                ship_df = ship_groups.get(mmsi)
                yield (mmsi, (traj_df, ship_df))

        return task_iter(), task_count

    def _load_data(self, glob_str):
        paths = Path(self.config["paths"]["in_folder"]).glob(glob_str)
        dfs = [pd.read_parquet(path, engine="pyarrow") for path in paths]
        df = pd.concat(dfs).reset_index(drop=True)
        return df

    def execut_task(self, mmsi, traj_df, ship_df):
        tp = TrajectoryProcessor(mmsi, traj_df, ship_df, shared_ship_info)

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
