from scipy.spatial import cKDTree
from utils.config import Config
import pandas as pd
import numpy as np
from preprocessing.utils.pipeline.function_inport import import_from_string


class ShipProcessor:

    def __init__(self, timestamp, df_nodes):
        self.config = Config().get("transform_edges")
        self.df_nodes = df_nodes
        self.df_edges = self.init_edges(timestamp)
        self.df_lookup = self.init_lookup()

    def init_edges(self, timestamp):
        gdf = self.df_nodes.copy().to_crs("epsg:32632")
        coords = list(zip(gdf.geometry.x, gdf.geometry.y))

        tree = cKDTree(coords)
        neighbors = tree.sparse_distance_matrix(
            tree, max_distance=self.config["max_edge_dist"], output_type="ndarray"
        )

        mmsis = gdf["mmsi"].to_numpy()
        mask = neighbors["i"] != neighbors["j"]
        num_edges = np.count_nonzero(mask)

        df_edges = pd.DataFrame(
            {
                "timestamp": [timestamp] * num_edges,
                "mmsi": mmsis[neighbors["i"]][mask],
                "mmsi_other": mmsis[neighbors["j"]][mask],
                "dist": neighbors["v"].round(2)[mask],
            }
        )
        return df_edges

    def init_lookup(self):
        left_df = self.df_edges[["mmsi", "mmsi_other"]].copy()
        right_df = pd.DataFrame(
            self.df_nodes[["mmsi", "lat", "lon", "calc_speed", "direction"]]
        )

        left_df["_row_order"] = np.arange(len(left_df))

        df_lookup = (
            left_df.merge(right_df, on="mmsi", how="left")
            .merge(
                right_df.add_suffix("_other"),
                left_on="mmsi_other",
                right_on="mmsi_other",
                how="left",
            )
            .sort_values("_row_order")
            .drop(columns="_row_order")
            .reset_index(drop=True)
        )
        return df_lookup

    def run_pipeline_steps(self, config_key):
        for step in self.config[config_key]:
            args = step.get("args", {}).copy()
            if "function" in step:
                operation = import_from_string(step["function"])
                args["ship_processor"] = self
            if "metric" in step:
                operation = self.add_feature
                args["metric"] = import_from_string(step["metric"])
            try:
                operation(**args)
            except Exception as error:
                raise Exception(f"during {step}: {error}")

    def add_feature(self, col_name, metric, arg_cols):
        args = [self.df_lookup[col] for col in arg_cols] + [
            self.df_lookup[col + "_other"] for col in arg_cols
        ]
        self.df_edges[col_name] = metric(*args)

    def get_df(self):
        return self.df_edges
