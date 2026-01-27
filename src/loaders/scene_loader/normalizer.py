import numpy as np
import pandas as pd

MINMAX = {
    'speed' : (0, 40, 0),
    'acc' : (-4, 4, 0),
    'length': (0, 400, 1), 
    'width': (0, 60, 1),
}
DEG_COS = set(['course', 'angular_difference', 'hour_of_day']) #, 'rel_bearing', 'course_diff'])

def normalize(nodes: pd.DataFrame, feature_cols) -> pd.DataFrame:
    if 'hour_of_day' in feature_cols:
        nodes["hour_of_day"] / 24 * 360

    norm_deg_cols = DEG_COS & set(feature_cols)
    print("normalizing:",  norm_deg_cols)
    for col in norm_deg_cols:
        rad = np.deg2rad(nodes[col])
        nodes[col + "_sin"] = np.sin(rad)
        nodes[col + "_cos"] = np.cos(rad)
        feature_cols += [col + "_sin", col + "_cos"]
        feature_cols.remove(col)
    nodes = nodes.drop(columns=norm_deg_cols)

    if "ship_group" in feature_cols:
        nodes["sailing"] = nodes["ship_group"] == "sailing"
        nodes["cargo"] = nodes["ship_group"] == "cargo"
        nodes["passenger"] = nodes["ship_group"] == "passenger"
        nodes = nodes.drop(columns=["ship_group"])
        feature_cols.remove("ship_group")
        feature_cols += ["sailing", "cargo", "passenger"]

    norm_min_max_cols = set(MINMAX.keys()) & set(feature_cols)
    print("normalizing:",  norm_min_max_cols)
    for col in norm_min_max_cols:
        min_val, max_val, use_log = MINMAX[col]
        if use_log:
            nodes[col] = np.log1p(nodes[col] - min_val) / np.log1p(max_val-min_val)
        else:
            nodes[col] = (nodes[col] - min_val) / (max_val-min_val)

    return nodes, feature_cols


    #'dist_to_land' : (0, 2000, 1), 
    #'dist_to_restricted_area' : (0, 2000, 1), 
    #'dist_to_ferry_route' : (0, 2000, 1), 
    #'water_depth': (0, 20, 0),
    # 'dist': (0, 2000, 1), 
    # 'rel_speed': (0, 40, 0), 
    # 'tcpa': (0, 3600, 1),
    # 'dcpa': (0, 2000, 1),
    #'x': (573663, 586057, 0), 
    #'y' : (6018805, 6035410, 0),
    #'rel_x' : (0, 100, 0),
    #'rel_y' : (0, 100, 0),