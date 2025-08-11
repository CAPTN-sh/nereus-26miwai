from pathlib import Path

import pandas as pd
import requests

URL = "https://api.myshiptracking.com/api/v2/vessel/bulk"


def load_file(path):
    if path.exists():
        return pd.read_parquet(path, engine="pyarrow")
    else:
        columns = ["mmsi", "ship_type", "to_bow", "to_stern", "to_port", "to_starboard"]
        return pd.DataFrame({col: pd.Series(dtype="int64") for col in columns})


def add_default(df, mmsi):
    df.loc[len(df)] = {
        "mmsi": mmsi,
        "ship_type": 0,
        "to_bow": 0,
        "to_stern": 0,
        "to_port": 0,
        "to_starboard": 0,
    }
    return df


def add_ship_info(df, entry):
    df.loc[len(df)] = {
        "mmsi": entry["mmsi"],
        "ship_type": entry["ais_type"],
        "to_bow": entry["size_a"],
        "to_stern": entry["size_b"],
        "to_port": entry["size_c"],
        "to_starboard": entry["size_d"],
    }
    return df


def get_missing_mmsi(df_ship, df_crawler):
    missing_mmsi = set(df_ship[df_ship["ship_type"].isna()]["mmsi"])
    missing_mmsi -= set(df_crawler["mmsi"])
    return missing_mmsi


def my_ship_tracking(api_key):
    path_crawler = Path("data/kiel/ais/3_features/ship_info_crawler.parquet")
    df_crawler = load_file(path_crawler)

    path_ship = Path("data/kiel/ais/3_features/ship_info.parquet")
    df_ship = load_file(path_ship)

    missing_mmsi = get_missing_mmsi(df_ship, df_crawler)

    invalid_mmsi = [mmsi for mmsi in missing_mmsi if mmsi >= 999999999]
    for mmsi in invalid_mmsi:
        df_crawler = add_default(df_crawler, mmsi)
    missing_mmsi -= set(invalid_mmsi)

    if len(missing_mmsi) == 0:
        print("nothing left to crawl")
        return

    while len(missing_mmsi) > 0:
        mmsi_selected = [int(mmsi) for mmsi in list(missing_mmsi)[:100]]

        mmsi_param = ",".join(str(mmsi) for mmsi in mmsi_selected)
        params = {
            "response": "extended",
            "mmsi": mmsi_param,
        }

        response = requests.get(URL, params=params, headers={"x-api-key": api_key})

        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")

        added_mmsi = []

        data = response.json()
        for entry in data.get("data", []):
            df_crawler = add_ship_info(df_crawler, entry)
            added_mmsi.append(entry["mmsi"])

        for mmsi in mmsi_selected:
            if mmsi in added_mmsi:
                continue
            df_crawler = add_default(df_crawler, mmsi)

        df_crawler.to_parquet(path_crawler)
        missing_mmsi -= set(mmsi_selected)
        print("left to crawl:", len(missing_mmsi))


if __name__ == "__main__":
    my_ship_tracking("bfg")  # add your api key
