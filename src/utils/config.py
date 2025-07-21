import yaml
import threading
import os
import shutil
from pathlib import Path


class Config:
    _instance = None
    _lock = threading.Lock()
    _path = Path("bin/config.yaml").resolve()
    _subconfigs = {}

    def __new__(cls, path=None):
        with cls._lock:
            if path is not None:
                cls._copy_to_bin(cls, path)
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def get(self, subconfig):
        try:
            return self._subconfigs[subconfig]
        except KeyError:
            raise KeyError(f"There is no subconfig named '{subconfig}'.")

    def _initialize(self):
        main_config = self._load_main_config()
        for name, path in main_config["config"].items():
            self._subconfigs[name] = self._load_config(path)
        self._load_folder(main_config["folder"])

    def _copy_to_bin(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        os.makedirs(self._path.parent, exist_ok=True)
        shutil.copy(path, self._path)

    def _load_main_config(self):
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"Config was not initialized")
        with open(self._path, "r") as f:
            return yaml.safe_load(f)

    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_folder(self, folders):
        self.folder = {}
        for name, path in folders.items():
            os.makedirs(Path(path).resolve(), exist_ok=True)
            self.folder[name] = Path(path).resolve()
