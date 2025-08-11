from pathlib import Path

import geopandas as gpd
import pandas as pd

from preprocessing.pipeline.pipeline import Pipeline
from preprocessing.steps.transform.edges.ship import ShipProcessor
from utils.config import Config


class EdgesPipeline(Pipeline):
    def __init__(self):
        self.config = Config().get("transform_edges")

    def load_tasks(self):
        path = Path(self.config["paths"]["in_folder"]) / "nodes.parquet"
        df = gpd.read_parquet(path)
        task_count = df["timestamp"].nunique()
        tasks = (
            (timestamp, (group_df,)) for timestamp, group_df in df.groupby("timestamp")
        )
        return tasks, task_count

    def execut_task(self, timestamp, df):
        sp = ShipProcessor(timestamp, df)
        for step in self.config["pipeline_steps"]:
            try:
                sp.run_pipeline_steps(step)
            except Exception as err:
                print(f"Error in step {step}: {err}")
                return
        try:
            return sp.get_df()
        except Exception as err:
            print(f"Error retrieving the trajectories: {err}")
            return

    def save_results(self, results):
        df = pd.concat(results, ignore_index=True)
        df.to_parquet(Path(self.config["paths"]["out_folder"]) / "edges.parquet")
