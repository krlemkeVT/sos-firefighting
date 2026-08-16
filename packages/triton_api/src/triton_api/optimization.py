from dataclasses import dataclass


@dataclass
class ObjectiveResult:
    value: float
    metrics: dict[str, float]
    feasible: bool = True