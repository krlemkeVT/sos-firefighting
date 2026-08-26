"""Aircraft sizing and performance models using OpenMDAO.

This package provides aircraft performance analysis — fuel burn per mission
segment, optimal cruise/loiter speeds, and mission profile generation —
outputting results in the SimulationInput format expected by the TRITON
wildfire simulation.
"""

from aircraft_sizing.sizer import DefaultAircraftSizer
from aircraft_sizing.spec_model import SpecAircraftModel, PerformanceSpec
from aircraft_sizing.performance import AircraftParams, Mission, MissionSegment
from aircraft_sizing.performance import run_mission, optimal_speeds, atmosphere

__all__ = [
    "DefaultAircraftSizer",
    "SpecAircraftModel",
    "PerformanceSpec",
    "AircraftParams",
    "Mission",
    "MissionSegment",
    "run_mission",
    "optimal_speeds",
    "atmosphere",
]
