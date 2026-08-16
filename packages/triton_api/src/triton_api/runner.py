"""Shared request and result contracts for wildfire runner workflows.

These dataclasses intentionally stay small and dependency-free so they can be
used by the runner, optimization layer, smoke examples, and tests without
pulling in the wildfire simulation itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AircraftFleetEntry:
    """Describe one aircraft entry that should appear in wildfire overwrites.

    The wildfire simulation allows multiple aircraft definitions inside one
    scenario. We therefore model fleet entries explicitly instead of assuming a
    single generated aircraft per run.
    """

    file_name: str
    agents_per_base: list[int]
    suppression_tactic: dict[str, Any]


@dataclass(slots=True)
class ScenarioRunRequest:
    """Describe one scenario that should be executed across one or more seeds."""

    run_id: str
    run_dir: Path
    scenario_name: str
    input_file: str
    baseline_overwrites_file: str
    fleet: list[AircraftFleetEntry]
    scenario_modifiers: dict[str, Any]
    seeds: list[int]
    fleet_acq_eur: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ScenarioRunResult:
    """Return standardized per-scenario execution details back to callers."""

    run_id: str
    scenario_name: str
    feasible: bool
    mean_moe: float | None
    per_seed: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    output_files: dict[str, Path] = field(default_factory=dict)
    errors: list[str] | None = None


@dataclass(slots=True)
class BatchRunRequest:
    """Describe a sequential batch of scenarios that belong to one study case."""

    batch_id: str
    batch_dir: Path
    scenarios: list[ScenarioRunRequest]
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class BatchRunResult:
    """Collect standardized results for one batch request."""

    batch_id: str
    results: list[ScenarioRunResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    output_files: dict[str, Path] = field(default_factory=dict)
    errors: list[str] | None = None
