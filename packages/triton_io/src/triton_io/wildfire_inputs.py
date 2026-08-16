"""Compatibility wrappers for the old wildfire input helper module.

Historically this module owned ad-hoc JSON writing helpers. The refactor moves
wildfire overwrite generation into :mod:`triton_io.wildfire_overwrites`, but we
keep this module as a thin facade so existing imports do not break abruptly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from triton_api.runner import AircraftFleetEntry
from triton_io.wildfire_overwrites import (
    build_overwrites,
    load_overwrites,
    write_json,
    write_overwrites,
)


def write_wildfire_inputs(
    run_dir: Path,
    scenario_config: dict[str, Any],
    aircraft_config: dict[str, Any],
) -> dict[str, Path]:
    """Preserve the legacy helper used by early monorepo scaffolding.

    This helper is still useful in tests and manual experiments, even though
    the newer runner path writes a generated ``overwrites.json`` instead of
    separate generic scenario/aircraft files.
    """

    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    scenario_json = write_json(input_dir / "scenario.json", scenario_config)
    aircraft_json = write_json(input_dir / "aircraft.json", aircraft_config)

    return {
        "scenario_json": scenario_json,
        "aircraft_json": aircraft_json,
        "input_dir": input_dir,
    }


__all__ = [
    "AircraftFleetEntry",
    "build_overwrites",
    "load_overwrites",
    "write_json",
    "write_overwrites",
    "write_wildfire_inputs",
]
