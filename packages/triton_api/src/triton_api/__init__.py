"""Convenience exports for the lightweight TRITON API package.

The goal of this package is to centralize shared contracts so the rest of the
monorepo can coordinate without circular imports.
"""

from triton_api.aircraft import AircraftDesignVariables, AircraftSizingResult
from triton_api.optimization import ObjectiveResult
from triton_api.runner import (
    AircraftFleetEntry,
    BatchRunRequest,
    BatchRunResult,
    ScenarioRunRequest,
    ScenarioRunResult,
)
from triton_api.simulation import SimulationInput, SimulationResult

__all__ = [
    "AircraftDesignVariables",
    "AircraftFleetEntry",
    "AircraftSizingResult",
    "BatchRunRequest",
    "BatchRunResult",
    "ObjectiveResult",
    "ScenarioRunRequest",
    "ScenarioRunResult",
    "SimulationInput",
    "SimulationResult",
]
