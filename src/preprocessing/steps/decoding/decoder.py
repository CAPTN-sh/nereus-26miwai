from pathlib import Path
from pyais import decode
from collections import defaultdict
from typing import List


class Decoder:
    """
    Decoder for files containing single- and multi-part AIS messages
    encoded by NMEA (National Marine Electronics Association).

    https://pypi.org/project/pyais/

    message example: !AIVDM,2,1,4,B,15MwkT1P37G?fl0EJbR0OwT0@MS,0*4E

    !AIVDM:
        This field indicates that the sentence is an AIS message.
    2,1:
        These fields indicate the total number of sentences in the message
        and the current sentence number.
    4:
        These fields contains the sequence number of multipart messages.
    B:
        This field indicates the communication channel being used.
    15MwkT1P37G?fl0EJbR0OwT0@MS:
        This field contains the payload of the message.
    """

    def __init__(self) -> None:
        self.raw_data = []
        self.successes = 0
        self.errors = 0
        self.buffer = defaultdict(list)
        self.timestamps = {}

    def decode_files(self, paths: List[Path]) -> List[dict]:
        """
        Decode AIS messages from a list of file paths.

        Args:
            paths (List[Path]): List of paths pointing to .nmea files.

        Returns:
            List[dict]: List of decoded AIS messages as dictionaries.
        """
        self.raw_data = []
        for path in paths:
            self._decode_nmea_file(path)
        return self.raw_data, self.successes, self.errors

    def _decode_nmea_file(self, path: Path) -> None:
        """
        Decode AIS messages from a songle file paths.

        Goes through the file line by line to extract AIVDM and DATE-TIME.
        Once all parts from a message are collected it is decoded and added
        to self.raw_data.

        Args:
            path (Path): Path pointing to a .nmea file.
        """
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

    def _decode_single_part(self, aivdm_line: str, timestamp: str) -> None:
        """
        Decode a single part AIS message and addes it to self.raw_data.

        Args:
            aivdm_line (str): Single part message to decode.
            timestamp (str): Timestamp when the message was received.
        """
        try:
            result = decode(aivdm_line).asdict()
            result["timestamp"] = timestamp
            self.raw_data.append(result)
            self.successes += 1
        except Exception:
            self.errors += 1

    def _decode_multi_part(self, fragment_id: str) -> None:
        """
        Decode a multi part AIS message
        by stitching it together into a "fake" single part message.

        args:
            fragment_id (str): The key where all message parts are stored.
        """
        try:
            full_message = "".join(
                p.split(",")[5]
                for p in sorted(
                    self.buffer[fragment_id],
                    key=lambda x: int(x.split(",")[2]),
                )
            )
            joined_line = self.buffer[fragment_id][0].split(",")
            joined_line[1:4] = ["1", "1", ""]
            joined_line[5] = full_message
            single_line = ",".join(joined_line)
            self._decode_single_part(single_line, self.timestamps[fragment_id])
        except Exception as e:
            print(str(type(e).__name__))
        finally:
            del self.buffer[fragment_id]
            del self.timestamps[fragment_id]
