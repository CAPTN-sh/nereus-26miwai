import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load available trajectory dates
out_folder = Path("C:/Users/Ben/Desktop/server/2_decoded")
paths = list(out_folder.glob("*_traj.parquet"))

dates_present = sorted([p.stem.split("_")[0] for p in paths])
dates_present = pd.to_datetime(dates_present, format="%Y%m%d")
date_range = pd.date_range(start=min(dates_present), end=max(dates_present), freq="D")

# Series for file existence
df_exist = pd.Series(1, index=dates_present)
df_exist = df_exist.reindex(date_range, fill_value=0)

# Monthly ticks
monthly_ticks = [i for i, d in enumerate(df_exist.index) if d.day == 1]
monthly_labels = [df_exist.index[i].strftime("%Y-%m") for i in monthly_ticks]

# File size
df_size = pd.read_parquet("C:/Users/Ben/Desktop/server/raw_info.parquet")
df_size["date"] = pd.to_datetime(df_size["date"])
df_size = df_size.set_index("date").reindex(date_range)

# Plot
fig, ax1 = plt.subplots(figsize=(15, 4))

# File existence as shaded area
df_exist.plot(kind="area", ax=ax1, color="green", alpha=0.3, label="File Exists")
ax1.set_ylabel("File Exists (shaded area)")
ax1.set_xticks(monthly_ticks)
ax1.set_xticklabels(monthly_labels, rotation=45, ha="right", fontsize=8)

# Secondary axis for size
ax2 = ax1.twinx()
df_size["size_MB"].plot(ax=ax2, color="blue", linewidth=2, label="Size (MB)")
ax2.set_ylabel("Size (MB)")

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.title("Trajectory File Availability and Size by Date")
plt.tight_layout()
plt.show()
