from dataclasses import dataclass
from typing import Any


@dataclass
class SimulationInput:
    parameters: dict[str, Any]


@dataclass
class SimulationResult:
    metrics: dict[str, float]
    artifacts: dict[str, Any] | None = None