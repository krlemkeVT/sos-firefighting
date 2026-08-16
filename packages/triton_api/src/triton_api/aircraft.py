from dataclasses import dataclass


@dataclass
class AircraftDesignVariables:
    variables: dict[str, float]


@dataclass
class AircraftSizingResult:
    feasible: bool
    metrics: dict[str, float]
    constraint_violations: dict[str, float]