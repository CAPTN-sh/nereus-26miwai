from collections import defaultdict
from pathlib import Path

import pandas as pd

from preprocessing.pipeline.pipeline import Pipeline
from preprocessing.steps.decoding.decoder import Decoder
from utils.config import Config


class DecodingPipeline(Pipeline):

    def __init__(self):
        self.config = Config().get("decode")

    def load_tasks(self):
        groups = defaultdict(list)
        for file in Path(self.config["paths"]["in_folder"]).glob("*.nmea.txt"):
            date_str = file.name[:8]
            groups[date_str].append(file)
        task_count = len(groups)

        def task_iter():
            for date_str, files in groups.items():
                yield date_str, (files,)

        return task_iter(), task_count

    def execut_task(self, date, paths):
        raw_data, successes, errors = Decoder().decode_files(paths)
        df_raw = pd.DataFrame(raw_data)
        if df_raw.empty:
            return (successes, errors)

        for name, schema in self.config["tables"].items():
            columns = list(schema["column_types"].keys())
            msg_types = schema["msg_types"]
            df = pd.DataFrame(raw_data)
            if "msg_type" not in df.columns:
                continue
            df = df[df["msg_type"].isin(msg_types)][columns]

            for c_name, c_type in schema["column_types"].items():
                if c_type == "datetime":
                    df[c_name] = pd.to_datetime(df[c_name], errors="coerce")
                elif c_type:
                    df[c_name] = df[c_name].astype(c_type, errors="ignore")
            df.to_parquet(
                Path(self.config["paths"]["out_folder"]) / f"{date}_{name}.parquet",
                index=False,
                engine="pyarrow",
            )
        return (successes, errors)

    def save_results(self, results):
        first_sum = sum(x[0] for x in results)
        second_sum = sum(x[1] for x in results)

        print("Successfully decoded:", first_sum)
        print("Failed to decode:", second_sum)
