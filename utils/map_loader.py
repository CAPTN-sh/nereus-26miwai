from utils.config import Config
import glob
import geopandas as gpd

DEFAULT_CRS = "EPSG:4326"


class MapLoader:
    _instance = None
    _maps: gpd.GeoDataFrame
    _map_layers: gpd.GeoDataFrame

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        map_folder = Config().folder["maps"]
        map_files = [
            gpd.read_file(path).to_crs(DEFAULT_CRS)
            for path in glob.glob(f"{map_folder}/*.geojson")
        ]
        self._maps = gpd.pd.concat(map_files, ignore_index=True).set_index("name")
        self._map_layers = self._maps.dissolve(by="layer")

    def get_features(self, layers=[], crs=DEFAULT_CRS) -> gpd.GeoDataFrame:
        if layers:
            return self._maps.to_crs(crs)[self._maps["layer"].isin(layers)]
        return self._maps

    def get_layers(self, crs=DEFAULT_CRS) -> gpd.GeoDataFrame:
        return self._map_layers.to_crs(crs)

    def get_layer(self, layer, crs=DEFAULT_CRS) -> gpd.GeoDataFrame:
        return self._map_layers.to_crs(crs).loc[layer]

    def total_bounds(self, crs=DEFAULT_CRS):
        return self._map_layers.to_crs(crs).total_bounds
