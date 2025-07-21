import json
from multiprocessing.synchronize import Lock as LockType
from multiprocessing.managers import DictProxy
from preprocessing.utils.ship_info_system.webcrawler import crawl_ship_info
import yaml
from utils.config import Config


class ShipInfo:
    def __init__(self, shared_db: DictProxy, lock: LockType):
        self.db = shared_db
        self.lock = lock
        self.folder = Config().folder["ship_db"]
        self.path = self.folder / "ship_db.json"
        self.ship_type = self._load_ship_type_dict()
        self._load_from_file()

    def _load_ship_type_dict(self):
        with open(self.folder / "ship_type_dict.yaml", "r") as f:
            return yaml.safe_load(f)

    def _load_from_file(self):
        if self.path.exists():
            with self.lock, open(self.path, "r") as f:
                data = json.load(f)
                self.db.update(data)

    def _save_to_file(self):
        with open(self.path, "w") as f:
            json.dump(dict(self.db), f, indent=2)

    def get_info(self, mmsi):
        if int(mmsi) in self.db:
            return self.db[int(mmsi)]

        with self.lock:
            if int(mmsi) in self.db:
                return self.db[int(mmsi)]

            err, msg, info = crawl_ship_info(mmsi)
            if err:
                return None

            self.db[int(mmsi)] = self.transform_info(info)
            self._save_to_file()
            return info

    def transform_info(self, info):
        info["ship_type"] = int(self.ship_type.get(info["ship_type"], 0))
        if info["length"] and info["width"]:
            info["to_bow"] = int(info["length"] / 2)
            info["to_stern"] = int(info["length"] / 2)
            info["to_port"] = int(info["width"] / 2)
            info["to_starboard"] = int(info["width"] / 2)
        else:
            info["to_bow"] = 0
            info["to_stern"] = 0
            info["to_port"] = 0
            info["to_starboard"] = 0
        return info
