import logging
from datetime import datetime
from pathlib import Path
import json
import os
import optuna
import fcntl

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

# ---------- JSONL logging (multi-process safe via file lock) ----------
def _append_jsonl_atomic(jsonl_path: Path, record: dict):
    """
    Append one JSON object per line to jsonl_path.
    Uses an advisory file lock (fcntl) so multiple GPU workers can write safely.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def trial_jsonl_callback(jsonl_path: Path):
    """
    Optuna callback factory: called after each trial finishes (success/pruned/fail).
    Appends a JSON record to a shared JSONL.
    """
    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        formatted_params = {k: round(v, 7) if isinstance(v, float) else v for k, v in trial.params.items()}
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trial_number": trial.number,
            "epochs_ran": trial.user_attrs.get("epochs_ran", None),
            "state": trial.state.name,
            "n_model_params": trial.user_attrs.get("n_model_params", None),
            "value": trial.value,
            "params": formatted_params,
            "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", None),
        }
        _append_jsonl_atomic(jsonl_path, record)

    return _cb