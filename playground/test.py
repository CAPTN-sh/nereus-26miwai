import pandas as pd
from pathlib import Path

data_folder = Path("/home/bbiesenbach/data/kiel/ais/3_features")

nodes = pd.read_parquet(data_folder / "nodes.parquet")
nodes["date"] = nodes["timestamp"].dt.date
counts = nodes["date"].value_counts().sort_index()
for d, c in counts.items():
    print(f"{d}: {c:,}")
