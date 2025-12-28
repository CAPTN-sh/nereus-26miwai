from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

KEEP = set([
    'timestamp', 'mmsi', 'traj_id', 'geometry', 'lon', 'lat', 'status', 'ship_type', 
    'crawled', 'mmsi_other', 'collision_risk', 'draught', 'time_diff'])
NUM = set([
    'speed', 'acc', 'dist_to_land', 'dist_to_restricted_area', 'dist_to_ferry_route', 
    'water_depth', 'density_all', 'density_sailing', 'density_cargo', 'density_other', 
    'density_passenger', "density", 'to_bow', 'to_stern', 'to_port', 'to_starboard', 
    "length", "width", 'dist', 'rel_speed', 'tcpa', 'dcpa'])
SIN_COS = set([
    'heading', 'course', 'angular_difference', 'course_of_rel_motion', 'course_diff', 
    'true_bearing', 'rel_bearing'])
CAT = set(['ais_class', 'rel_bearing_cat'])

def to_sin_cos(df, cols):
    for col in cols:
        rad = np.deg2rad(df[col])
        df[col + "_sin"] = np.sin(rad)
        df[col + "_cos"] = np.cos(rad)
    new_cols = set([c + "_sin" for c in cols] + [c + "_cos" for c in cols])
    return df, new_cols

def normalize(nodes, normalizer_path, cache_name):
    cols = set(nodes.columns)
    nodes, new_sin_cos_cols = to_sin_cos(nodes, SIN_COS & cols)
    num_cols = sorted((NUM & cols) | new_sin_cos_cols)
    cat_cols = sorted((CAT & cols))
    keep_cols = sorted((KEEP & cols))

    if normalizer_path is None:
        normalizer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop",
        )
        normalizer.fit(nodes)
        normalizer_path = Path("data/cache") / f"{cache_name}_normalizer.pkl"
        joblib.dump(normalizer, normalizer_path)
    else:
        normalizer = joblib.load(normalizer_path)

    X_df = pd.DataFrame(
        normalizer.transform(nodes), 
        index=nodes.index, 
        columns=normalizer.get_feature_names_out()
    )

    nodes = pd.concat([nodes[keep_cols].copy(), X_df], axis=1)
    print(nodes.head())
    print(nodes.mean())
    print(nodes.min())
    print(nodes.max())
    return nodes