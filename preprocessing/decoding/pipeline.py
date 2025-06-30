from preprocessing.decoding.decoder import Decoder
from pathlib import Path
import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict


class DecodingPipeline:

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def run(self):
        groups = self.group_files_by_date()
        with ProcessPoolExecutor() as executor:
            tasks = {
                executor.submit(self.decode_file, date, files): date
                for date, files in groups.items()
            }

            for task in tqdm(
                as_completed(tasks), total=len(tasks), desc="Processing dates"
            ):
                try:
                    task.result()
                except Exception as e:
                    print(f"Error processing {tasks[task]}: {e}")

    def group_files_by_date(self):
        groups = defaultdict(list)
        for file in Path(self.config["in_folder"]).glob("*.nmea.txt"):
            date_str = file.name[:8]
            groups[date_str].append(file)
        return groups

    def decode_file(self, date, paths):
        decoder = Decoder(self.config)
        decoder.decode_date(date, paths)
