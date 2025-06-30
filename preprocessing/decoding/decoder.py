from pathlib import Path
from pyais import decode
from collections import defaultdict
import pandas as pd
from typing import List


class Decoder:

    def __init__(self, config):
        self.config = config
        self.raw_data = []
        self.buffer = defaultdict(list)
        self.timestamps = {}

    def decode_date(self, date: str, paths: List[Path]):
        for path in paths:
            self._decode_nmea_file(path)
        self._extract_and_save_table(date)

    def _decode_nmea_file(self, path: Path):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        while i < len(lines) - 1:
            aivdm_line = lines[i]
            date_line = lines[i + 1]
            if not (
                aivdm_line.startswith("!AIVDM") and date_line.startswith("!DATE-TIME")
            ):
                i += 1
                continue
            i += 2
            try:
                parts = aivdm_line.split(",")
                n_total = int(parts[1])
                fragment_id = parts[3]
                date_str = f"{path.name[:4]}-{path.name[4:6]}-{path.name[6:8]}"
                timestamp = f"{date_str}T{date_line.split(',')[1].strip()}"

                if n_total == 1:
                    self._decode_single_part(aivdm_line, timestamp)
                else:
                    self.buffer[fragment_id].append(aivdm_line)
                    self.timestamps[fragment_id] = timestamp

                    if len(self.buffer[fragment_id]) == n_total:
                        self._decode_multi_part(fragment_id)

            except Exception as e:
                print(str(type(e).__name__))

    def _decode_single_part(self, aivdm_line, timestamp):
        result = decode(aivdm_line).asdict()
        result["timestamp"] = timestamp
        self.raw_data.append(result)

    def _decode_multi_part(self, fragment_id):
        try:
            full_message = "".join(
                p.split(",")[5]
                for p in sorted(
                    self.buffer[fragment_id],
                    key=lambda x: int(x.split(",")[2]),
                )
            )
            joined_line = self.buffer[fragment_id][0].split(",")
            joined_line[1:4] = ["1", "1", ""]  # Fake as single part
            joined_line[5] = full_message
            single_line = ",".join(joined_line)
            self._decode_single_part(single_line, self.timestamps[fragment_id])
        except Exception as e:
            print(str(type(e).__name__))
        finally:
            del self.buffer[fragment_id]
            del self.timestamps[fragment_id]

    def _extract_and_save_table(self, date):
        for name, schema in self.config["tables"].items():
            columns = list(schema["column_types"].keys())
            df = pd.DataFrame(self.raw_data)[columns]
            df.dropna(subset=schema["key_columns"], inplace=True)

            for c_name, c_type in schema["column_types"].items():
                if c_type == "datetime":
                    df[c_name] = pd.to_datetime(df[c_name], errors="coerce")
                elif c_type:
                    df[c_name] = df[c_name].astype(c_type, errors="ignore")
            out_path = Path(self.config["out_folder"]) / f"{date}_{name}.parquet"
            df.to_parquet(out_path, index=False, engine="pyarrow")
