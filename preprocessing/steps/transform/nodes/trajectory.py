import pandas as pd
from pandas import DataFrame
from movingpandas import Trajectory, TrajectoryCollection
from preprocessing.utils.pipeline.function_inport import import_from_string
from preprocessing.utils.df_helper import drop_duplicates, to_GeoDataFrame
from utils.config import Config
from preprocessing.utils.ship_info_system.ship_info import ShipInfo
from geopandas import GeoDataFrame


class TrajectoryProcessor:
    """
    The TrajectoryProcessor stores the trajectories of one vessel (mmsi).
    Here all preprocessing steps for the nodes pipeline are applied to the trajectories.
    """

    def __init__(
        self, mmsi: int, traj_df: DataFrame, ship_df: DataFrame, ship_dict: ShipInfo
    ) -> None:
        self.config = Config().get("transform_nodes")
        self.trajectories = self.init_trajectories(mmsi, traj_df, ship_df, ship_dict)

    def init_trajectories(
        self, mmsi: int, traj_df: DataFrame, ship_df: DataFrame, ship_dict: ShipInfo
    ) -> TrajectoryCollection:
        """
        Initialize the TrajectoryCollection for a given MMSI.

        Args:
            mmsi (int): Maritime Mobile Service Identity.
            traj_df (DataFrame): Raw trajectory data.
            ship_df (DataFrame): Raw ship metadata.
            ship_dict (ShipInfo): External source for ship metadata.

        Returns:
            TrajectoryCollection:
                The trajectory data and ship metadata turned into a Trajectory object.
        """
        gdf = to_GeoDataFrame(drop_duplicates(traj_df))
        traj = []
        if len(gdf) > 1:
            info = self.get_ship_info(mmsi, ship_df, ship_dict)
            for col, value in info.items():
                gdf[col] = value
            traj = [Trajectory(gdf, traj_id=0, obj_id=mmsi, t="timestamp")]
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

    def get_ship_info(
        self, mmsi: int, ship_df: DataFrame, ship_dict: ShipInfo
    ) -> dict[str, list]:
        """
        Filles missing data from the AIS messages with data from the web.

        Args:
            mmsi (int): Maritime Mobile Service Identity.
            ship_df (DataFrame): Raw ship metadata.
            ship_dict (ShipInfo): External source for ship metadata.

        Returns:
            dict[str, list]: Dict with static ship info like ship_type.
        """
        keys = ["mmsi", "ship_type", "to_bow", "to_stern", "to_port", "to_starboard"]
        web_info = ship_dict.get_info(mmsi)
        info = {k: web_info[k] for k in keys if k in web_info}

        keys = ["to_bow", "to_stern", "to_port", "to_starboard"]
        if ship_df is not None and not ship_df.empty:
            for k in keys:
                v = ship_df[k].max()
                if pd.notna(v) and v > 0:
                    info[k] = v
        return info

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
            traj for traj in self.trajectories.trajectories if len(traj.df) >= min_obs
        ]
        self.trajectories = TrajectoryCollection(
            valid_traj, traj_id_col="traj_id", obj_id_col="mmsi", t="timestamp"
        )
