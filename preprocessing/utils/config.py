import yaml

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_config(path):
    with open(BASE_DIR / path, "r") as f:
        config = yaml.safe_load(f)

    for key, rel_path in config["paths"].items():
        config["paths"][key] = BASE_DIR / rel_path

    return config
