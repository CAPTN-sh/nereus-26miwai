import logging
from datetime import datetime
from pathlib import Path


def logger(file_prefix: str = "run", log_dir: str = ".logs", level=logging.INFO):
    """
    Configure logging to output both to console and to a timestamped log file.
    Can be called once from the main entrypoint.
    """
    if logging.getLogger().handlers:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = Path(log_dir) / f"{file_prefix}_{timestamp}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )

    logging.info("Logger initialized. Writing logs to %s", logfile)
