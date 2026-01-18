import pandas as pd
import numpy as np

from utils.config import DATA_FOLDER_PATH


df = pd.read_parquet(DATA_FOLDER_PATH / "ais/4_features/fh/kiel/fh_kiel_val_ship_features.parquet")
ship = pd.read_parquet(DATA_FOLDER_PATH / "ship_db/ship_db.parquet")

df = df.merge(ship, on="mmsi")


traj = df.groupby("traj_id")
traj_has_entry = traj["is_entry"].any()

def get_q(traj):
    traj_len = traj.value_counts().values

    O = 12  # observation steps
    F = 12  # minimum future steps
    qs = [0.05, 0.15, 0.25, 0.50, 0.75, 0.85, 0.95]

    max_futures = []

    for T in traj_len:
        first_start = O
        last_start = T-F
        if last_start < first_start:
            continue

        for t_last_obs in range(first_start, last_start + 1):
            max_futures.append(T - t_last_obs)

    max_futures = np.array(max_futures)
    q_vals = np.quantile(max_futures, qs) * 5 / 60

    return pd.Series(q_vals, index=[f"{int(q*100)}%" for q in qs]).rename("H (minutes)")

print("all")
print(get_q(df["traj_id"]))

for l, g in df.groupby("ship_group"):
    print(l)
    print(get_q(g["traj_id"]))

"""

for ship_group, g in df.groupby("ship_group"):
    traj = g.groupby("traj_id")
    traj_has_entry = traj["is_entry"].any()

    n_traj = traj_has_entry.size
    n_with = traj_has_entry.sum()
    n_without = n_traj - n_with

    print(ship_group)
    print("with", n_with, "without", n_without, " with pct", n_with/n_traj * 100)
"""