from pathlib import Path

DATA_FOLDER_PATH = Path("/data/projects/ship_tracker/assets")

AIS_FOLDER_PATH = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"
MAP_FOLDER_PATH = DATA_FOLDER_PATH / "maps/2_standardized/fh_10/kiel"
SHIP_DB_PATH = DATA_FOLDER_PATH / "ship_db/ship_db.parquet"

AIS_SOURCE = "fh" # "fh" or "dma"

TRAIN_BBOX = [10.12, 54.31, 10.33, 54.46] # Training is done on Kiel
DEFAULT_CRS = "EPSG:4326"
AREA_CRS = "EPSG:3035"
METER_CRS = "EPSG:25832"

STEP_SIZE = 10 # interpolation step size from preprocessing
STEPS_PER_MINUTE = 60 // STEP_SIZE