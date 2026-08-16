"""Helpers for running an Optuna study with the refactored objective."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from triton_optimization.objective import objective


def _setup_logging(run_root: Path) -> None:
    """Mirror the existing worker/study logging setup in package form."""

    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    worker_id = os.environ.get("SLURM_PROCID", str(os.getpid()))
    log_path = log_dir / f"worker_{worker_id}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info("Logging to %s", log_path)


def _build_storage(storage_url: str) -> Any:
    """Build the Optuna storage object, including PostgreSQL pooling tweaks."""

    import optuna

    if storage_url.startswith("postgres"):
        from optuna.storages import RDBStorage
        from sqlalchemy.pool import NullPool

        return RDBStorage(
            url=storage_url,
            skip_table_creation=True,
            engine_kwargs={
                "poolclass": NullPool,
                "pool_pre_ping": True,
            },
        )

    return storage_url


def run_study() -> None:
    """Run the refactored Optuna study using environment-based configuration."""

    import optuna

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        seed=42,
        constant_liar=True,
    )

    run_root = Path(os.environ["OPTUNA_RUN_DIR"]).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_root)

    study = optuna.create_study(
        study_name=os.environ["OPTUNA_STUDY_NAME"],
        storage=_build_storage(os.environ["OPTUNA_STORAGE"]),
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(
        objective,
        n_trials=int(os.environ.get("OPTUNA_TOTAL_TRIALS", 500)),
    )
