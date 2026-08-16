"""Worker entrypoint for distributed Optuna execution."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from triton_optimization.objective import objective


def _setup_logging(run_root: Path, worker_id: str) -> None:
    """Set up per-worker logging so scheduler jobs stay readable."""

    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_{worker_id}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info("Logging to %s", log_path)


def run_worker() -> None:
    """Connect to an existing Optuna study and execute trials in this worker."""

    import optuna

    run_root = Path(os.environ["OPTUNA_RUN_DIR"]).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    worker_id = (
        os.environ.get("SLURM_PROCID")
        or os.environ.get("SLURM_ARRAY_TASK_ID")
        or os.environ.get("PBS_ARRAY_INDEX")
        or os.environ.get("HOSTNAME")
        or "0"
    )
    _setup_logging(run_root, str(worker_id))

    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    study = optuna.create_study(
        direction="maximize",
        study_name=os.environ["OPTUNA_STUDY_NAME"],
        storage=os.environ["OPTUNA_STORAGE"],
        load_if_exists=True,
    )

    n_trials = int(os.environ.get("OPTUNA_N_TRIALS", "10"))
    logging.info(
        "Worker %s starting: study=%s trials=%s",
        worker_id,
        os.environ["OPTUNA_STUDY_NAME"],
        n_trials,
    )
    study.optimize(objective, n_trials=n_trials)
    logging.info("Worker %s done.", worker_id)
