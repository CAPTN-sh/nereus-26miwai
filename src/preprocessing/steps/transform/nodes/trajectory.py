import pandas as pd
from movingpandas import Trajectory, TrajectoryCollection
from preprocessing.utils.pipeline.function_inport import import_from_string
from preprocessing.utils.df_transformer import drop_duplicates, to_GeoDataFrame
from utils.config import Config
from preprocessing.utils.ship_info_system.ship_info import ShipInfo


class TrajectoryProcessor:

    def __init__(self, mmsi, traj_df, ship_df, ship_dict: ShipInfo):
        self.config = Config().get("transform_nodes")
        self.mmsi = mmsi
        self.ship_df = ship_df
        self.ship_dict = ship_dict
        self.trajectories = self.init_trajectories(traj_df)

    def init_trajectories(self, traj_df):
        gdf = to_GeoDataFrame(drop_duplicates(traj_df))
        traj = []
        if len(gdf) > 1:
            traj = [Trajectory(gdf, traj_id=0, obj_id=self.mmsi, t="timestamp")]
        trajectories = TrajectoryCollection(
            traj,
            traj_id_col="traj_id",
            obj_id_col="mmsi",
            t="timestamp",
        )
        return trajectories

    def get_df(self):
        if not self.trajectories:
            return
        df = pd.concat(traj.df.reset_index() for traj in self.trajectories)
        for col, value in self.get_ship_info().items():
            df[col] = value
        return df

    def get_ship_info(self):
        keys = ["mmsi", "ship_type", "to_bow", "to_stern", "to_port", "to_starboard"]
        info = self.ship_dict.get_info(self.mmsi)
        info = {k: info[k] for k in keys if k in info}

        keys = ["to_bow", "to_stern", "to_port", "to_starboard"]
        if self.ship_df is not None and not self.ship_df.empty:
            for k in keys:
                v = self.ship_df[k].max()
                if pd.notna(v) and v > 0:
                    info[k] = v
        return info

    def run_pipeline_steps(self, config_key):
        for step in self.config[config_key]:
            args = step.get("args", {}).copy()
            for key, value in args.items():
                if key == "units" and isinstance(value, list):
                    args["units"] = tuple(value)

            if "method" in step:
                operation = getattr(self.trajectories, step["method"])
            if "function" in step:
                operation = import_from_string(step["function"])
                args["trajectories"] = self.trajectories
            try:
                self.trajectories = operation(**args)
            except Exception as error:
                raise Exception(f"during {operation}: {error}")
            self.validate_trajectories()

    def validate_trajectories(self):
        self.trajectories = TrajectoryCollection(
            [traj for traj in self.trajectories.trajectories if len(traj.df) >= 2],
            traj_id_col="traj_id",
            obj_id_col="mmsi",
            t="timestamp",
        )
