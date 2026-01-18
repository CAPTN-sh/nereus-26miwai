import pandas as pd
import numpy as np

from utils.config import DATA_FOLDER_PATH

def calculate_quintile_bins(series, dead_zone, precision=3, name="Feature"):
    """
    Calculates 11 symmetric bins using 20% quintiles of the maneuvers.
    Includes comma-separated formatting for easy copy-pasting.
    """
    # 1. Isolate the data outside the noise floor
    abs_series = series.abs()
    outside_dz = series[abs_series > dead_zone]
    
    # 2. Calculate Quintiles (20, 40, 60, 80)
    maneuver_magnitudes = outside_dz.abs()
    quantiles = np.percentile(maneuver_magnitudes, [20, 40, 60, 80])
    p20, p40, p60, p80 = quantiles

    # 3. Construct the 10 Symmetric Internal Walls
    edges = np.array([
        -p80, -p60, -p40, -p20, -dead_zone, 
        dead_zone, p20, p40, p60, p80
    ])
    
    # 4. Round and ensure uniqueness
    inner_edges = np.unique(np.round(edges, precision))

    # --- Formatting for Copy-Paste ---
    formatted_list = ", ".join([f"{val:.{precision}f}" for val in inner_edges])

    print(f"\n--- Analysis for {name} (11 Bins) ---")
    print(f"Maneuver Quintiles: 20%:{p20:.4f}, 40%:{p40:.4f}, 60%:{p60:.4f}, 80%:{p80:.4f}")
    print(f"Final Rounded Inner Edges (for Thesis/Code):")
    print(f"[{formatted_list}]")
    print(f"Vocabulary Size (Tokens): {len(inner_edges) + 1}")
    
    return inner_edges

# --- Load and Process ---
df = pd.read_parquet(DATA_FOLDER_PATH / "ais/4_features/fh/kiel/fh_kiel_train_ship_features.parquet")

rot_edges = calculate_quintile_bins(df["angular_difference"], dead_zone=0.5, precision=1, name="RoT")
acc_edges = calculate_quintile_bins(df["acc"], dead_zone=0.005, precision=3, name="Acceleration")

print("\n" + "="*50)
print("PYTHON-READY CONSTANTS")
print("="*50)
print(f"ROT_EDGES = np.array([{', '.join([f'{x:.1f}' for x in rot_edges])}])")
print(f"ACC_EDGES = np.array([{', '.join([f'{x:.3f}' for x in acc_edges])}])")