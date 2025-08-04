import pandas as pd
from pandas import DataFrame
from movingpandas import Trajectory, TrajectoryCollection
from preprocessing.utils.pipeline.function_inport import import_from_string
from preprocessing.utils.df_helper import drop_duplicates, to_GeoDataFrame
from utils.config import Config
from geopandas import GeoDataFrame


class TrajectoryProcessor:
    """
    The TrajectoryProcessor stores the trajectories of one vessel (mmsi).
    Here all preprocessing steps for the nodes pipeline are applied to the trajectories.
    """

    def __init__(self, mmsi: int, traj_df: DataFrame, ship_df: DataFrame) -> None:
        self.config = Config().get("transform_nodes")
        self.trajectories = self.init_trajectories(mmsi, traj_df, ship_df)

    def init_trajectories(
        self, mmsi: int, traj_df: DataFrame, ship_df: DataFrame
    ) -> TrajectoryCollection:
        """
        Initialize the TrajectoryCollection for a given MMSI.

        Args:
            mmsi (int): Maritime Mobile Service Identity.
            traj_df (DataFrame): Raw trajectory data.
            ship_df (DataFrame): Ship metadata.

        Returns:
            TrajectoryCollection:
                The trajectory data and ship metadata turned into a Trajectory object.
        """

        traj_df = traj_df.merge(ship_df, on="mmsi", how="left")
        gdf = to_GeoDataFrame(drop_duplicates(traj_df))
        traj = []
        if len(gdf) > 1:
            traj.append(Trajectory(gdf, traj_id=0, obj_id=mmsi, t="timestamp"))
        trajectories = TrajectoryCollection(
            traj,
            traj_id_col="traj_id",
            obj_id_col="mmsi",
            t="timestamp",
        )
        return trajectories

    def get_df(self) -> GeoDataFrame:
        """
        Returns the trajectories as a GeoDataFrame.
        Should be called after all preprocessing steps are done.

        Returns:
            GeoDataFrame
        """
        if not self.trajectories:
            return
        df = pd.concat(traj.df.reset_index() for traj in self.trajectories)
        return df

    def run_pipeline_steps(self, config_key: str) -> None:
        """
        Runs a pipeline step defined in config.

        "method" is called on the TrajectoryCollection
        "function" is imported and the TrajectoryCollection passed as an argument.

        Args:
            config_key (str): The key to the stp defined in the config.
        """
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

    def validate_trajectories(self) -> None:
        """
        Trajectories with few observations are droped after every pipeline step.
        """
        min_obs = self.config["global_args"]["min_obs"]
        valid_traj = [
            traj
            for traj in self.trajectories.trajectories
            if len(traj.df) >= min_obs and not traj.df.dropna(how="all", axis=1).empty
        ]
        self.trajectories = TrajectoryCollection(
            valid_traj, traj_id_col="traj_id", obj_id_col="mmsi", t="timestamp"
        )
