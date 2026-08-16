"""Helpers for building wildfire overwrite files from runner contracts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from triton_api.runner import AircraftFleetEntry


def write_json(path: Path, data: dict[str, Any]) -> Path:
    """Write JSON with a stable, human-readable format for debugging."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_overwrites(path: Path) -> dict[str, Any]:
    """Load a baseline overwrite file from disk.

    The runner resolves filenames to absolute paths before calling this helper,
    which keeps this IO module focused on reading and transforming JSON.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _fleet_entry_to_dict(entry: AircraftFleetEntry) -> dict[str, Any]:
    """Translate a shared API dataclass into wildfire overwrite JSON."""

    return {
        "file_name": entry.file_name,
        "agents_per_base": list(entry.agents_per_base),
        "suppression_tactic": deepcopy(entry.suppression_tactic),
    }


def _recursive_merge(
    base_value: dict[str, Any],
    modifiers: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge modifier dictionaries into a baseline structure.

    This is intentionally conservative: lists and scalar values are replaced
    outright, while nested dictionaries are merged key-by-key.
    """

    merged = deepcopy(base_value)
    for key, value in modifiers.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _recursive_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_overwrites(
    baseline_overwrites_path: Path,
    fleet: list[AircraftFleetEntry],
    scenario_modifiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a generated overwrite document for one scenario run.

    The current wildfire optimization flow replaces the full ``agents`` list
    with a combination of baseline and generated aircraft entries. We preserve
    that behavior exactly here instead of trying to patch individual items.
    """

    overwrites = load_overwrites(baseline_overwrites_path)
    overwrites["agents"] = [_fleet_entry_to_dict(entry) for entry in fleet]

    if scenario_modifiers:
        overwrites = _recursive_merge(overwrites, scenario_modifiers)

    return overwrites


def write_overwrites(path: Path, overwrites: dict[str, Any]) -> Path:
    """Write a generated wildfire overwrite file to the run directory."""

    return write_json(path, overwrites)
