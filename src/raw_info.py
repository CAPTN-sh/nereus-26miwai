import pandas as pd
from pathlib import Path
from utils.config import Config

# Path to the folder
config = Config("src/preprocessing/configs/_main.yaml")
folder = Path(config.get("decode")["paths"]["in_folder"])

# List all *.nmea.txt files
files = list(folder.glob("*.nmea.txt"))

# Extract date and file size
data = []
for f in files:
    try:
        date_str = f.name.split("-")[0]  # e.g., '20220501'
        date = pd.to_datetime(date_str, format="%Y%m%d")
        size = f.stat().st_size  # size in bytes
        data.append((date, size))
    except Exception as e:
        print(f"Skipping {f.name}: {e}")

# Build DataFrame
df = pd.DataFrame(data, columns=["date", "size_bytes"])

# Group and sum per date
df_summary = df.groupby("date")["size_bytes"].sum().reset_index()

# Optional: convert to MB or GB
df_summary["size_MB"] = df_summary["size_bytes"] / 1e6

# Save to file
df_summary.to_parquet("/home/bbiesenbach/data/kiel/ais/raw_info.parquet", index=False)

# Print preview
print(df_summary.head())
