import geopandas as gpd
from shapely import Point


def drop_duplicates(df):
    df["non_null_count"] = df.notnull().sum(axis=1)
    df = df.sort_values("non_null_count", ascending=False)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.drop("non_null_count", axis=1)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    return df


def to_GeoDataFrame(df, index=None):
    df["geometry"] = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df)
    gdf.set_geometry("geometry", inplace=True)
    gdf.set_crs("epsg:4326", inplace=True)
    if index is not None:
        gdf.set_index(index, inplace=True)
    return gdf
