"""Helpers for locating and reading wildfire JSON output files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _raise_missing_output(path: Path, description: str) -> None:
    """Raise a detailed error that points callers to nearby output candidates."""

    nearby = sorted(candidate.name for candidate in path.parent.glob("*.json"))
    raise FileNotFoundError(
        f"Missing {description}: {path}. Nearby JSON files: {nearby}"
    )


def read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON file with a single shared utility for callers and tests."""

    if not path.exists():
        _raise_missing_output(path, "JSON output file")
    return json.loads(path.read_text(encoding="utf-8"))


def get_simulation_output_path(out_base: Path) -> Path:
    """Return the expected simulation summary file for a wildfire run."""

    path = out_base.parent / f"{out_base.name}_simulation.json"
    if not path.exists():
        _raise_missing_output(path, "simulation output file")
    return path


def find_agent_output_files(out_base: Path) -> dict[str, Path]:
    """Locate per-agent JSON outputs emitted by the wildfire simulation.

    The SoSID writer names agent files with a shared prefix plus an
    ``_agents_<AgentType>.json`` suffix. Returning them as a mapping keeps the
    result easy to serialize and introspect.
    """

    prefix = f"{out_base.name}_agents_"
    matches = sorted(out_base.parent.glob(f"{prefix}*.json"))
    return {
        match.stem.removeprefix(prefix): match
        for match in matches
    }


def collect_output_files(out_base: Path) -> dict[str, Path]:
    """Collect the main simulation file and any agent files for one run."""

    output_files: dict[str, Path] = {
        "out_base": out_base,
        "simulation_json": get_simulation_output_path(out_base),
    }

    for agent_name, path in find_agent_output_files(out_base).items():
        output_files[f"agent_{agent_name}"] = path

    return output_files
