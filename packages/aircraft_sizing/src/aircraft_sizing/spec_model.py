"""Spec-based aircraft performance model — no sizing, no mission.

Takes performance specs directly as inputs (fuel rates, speeds, weights,
profile parameters) and outputs the SimulationInput-compatible dict that
the TRITON wildfire simulation expects.

Unlike the sizer (which computes fuel burn from aerodynamic parameters via
OpenConcept/OpenMDAO mission analysis), this model is a straight pass-through:
what you put in is what you get out. Use it when you already have measured or
published performance data and just need it in the sim's JSON format.

Branch: feature/performance-spec-model
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any


# ──────────────────────────────────────────────
# Performance spec dataclass
# ──────────────────────────────────────────────

@dataclass
class PerformanceSpec:
    """Direct performance specifications — no computation, just inputs.

    All values are provided directly; the model packages them into the
    wildfire simulation's aircraft JSON format without running any
    aerodynamic or mission analysis.
    """

    # ── Mass & geometry ──
    mtom: float                          # max takeoff mass (kg)
    empty_mass: float                    # operating empty mass (kg)
    payload: float = 0.0                # payload (kg)
    span: float = 0.0                   # wingspan (m)

    # ── Firefighting specs ──
    flow_rate: float = 0.0              # water drop flow rate (L/s or m³/s)
    can_scoop: bool = False             # can scoop water from lakes?
    scooping_distance: float = 0.0     # distance needed to scoop (m)

    # ── Sim metadata ──
    icon: str = "seaplane.svg"
    takeoff_landing_type: str = "runway"  # "runway" or "vertical"
    autonomous: bool = False

    # ── Propulsion: fuel & rates ──
    total_propellant: float = 0.0       # total fuel capacity (kg)
    reserve_propellant: float = 0.0     # reserve fuel (kg)
    refueling_rate: float = 7.7        # refueling rate (kg/s)

    # Per-segment fuel consumption rates (kg/s)
    # If a segment rate is not provided, defaults to 0.0
    taxi_out_fc: float = 0.0
    taxi_in_fc: float = 0.0
    takeoff_fc: float = 0.0
    transition_fc: float = 0.0
    retransition_fc: float = 0.0
    cruise_fc: float = 0.0
    cruise_climb_fc: float = 0.0
    cruise_descent_fc: float = 0.0
    landing_fc: float = 0.0
    loiter_fc: float = 0.0

    # ── Mission profile parameters ──
    taxi_out_duration: float = 120.0
    taxi_in_duration: float = 120.0
    transition_duration: float = 0.0
    retransition_duration: float = 0.0

    takeoff_altitude: float = 457.2     # m (1,500 ft)
    takeoff_climb_rate: float = 3.78    # m/s
    takeoff_ground_speed: float = 18.25 # m/s

    cruise_altitude: float = 3000.0     # m
    cruise_speed: float = 80.0          # m/s
    cruise_climb_rate: float = 8.85     # m/s
    cruise_climb_ground_speed: float = 84.55  # m/s
    cruise_descent_rate: float = 8.85   # m/s
    cruise_descent_ground_speed: float = 84.55  # m/s

    landing_altitude: float = 457.2     # m
    landing_descent_rate: float = 5.0   # m/s
    landing_ground_speed: float = 18.25 # m/s

    loiter_speed: float = 60.0          # m/s


# ──────────────────────────────────────────────
# Spec presets — known aircraft with published performance
# ──────────────────────────────────────────────

SPECS = {
    "cl415": dict(
        mtom=19890, empty_mass=12880, payload=6200, span=28.38,
        flow_rate=1.2, can_scoop=True, scooping_distance=410,
        icon="seaplane.svg", takeoff_landing_type="runway",
        total_propellant=4650, reserve_propellant=697.5, refueling_rate=7.7,
        taxi_out_fc=0.091, taxi_in_fc=0.091,
        takeoff_fc=0.495,
        cruise_fc=0.226, cruise_climb_fc=0.41, cruise_descent_fc=0.052,
        landing_fc=0.016, loiter_fc=0.232,
        cruise_altitude=1500, cruise_speed=72.2, loiter_speed=55.0,
        takeoff_altitude=457.2, landing_altitude=457.2,
    ),
    "dhc515": dict(
        mtom=20547, empty_mass=12995, payload=7000, span=28.6,
        flow_rate=1.2, can_scoop=True, scooping_distance=410,
        icon="seaplane.svg", takeoff_landing_type="runway",
        total_propellant=4626, reserve_propellant=693.9, refueling_rate=7.7,
        taxi_out_fc=0.091, taxi_in_fc=0.091,
        takeoff_fc=0.495,
        cruise_fc=0.226, cruise_climb_fc=0.41, cruise_descent_fc=0.052,
        landing_fc=0.016, loiter_fc=0.232,
        cruise_altitude=3000, cruise_speed=72.2, loiter_speed=55.0,
        takeoff_altitude=457.2, landing_altitude=457.2,
    ),
    "at802f": dict(
        mtom=7257, empty_mass=3062, payload=3000, span=18.04,
        flow_rate=1.2, can_scoop=False, scooping_distance=0,
        icon="seaplane.svg", takeoff_landing_type="runway",
        total_propellant=933, reserve_propellant=140.0, refueling_rate=7.7,
        taxi_out_fc=0.007, taxi_in_fc=0.007,
        takeoff_fc=0.101,
        cruise_fc=0.066, cruise_climb_fc=0.066, cruise_descent_fc=0.022,
        landing_fc=0.005, loiter_fc=0.046,
        cruise_altitude=2438, cruise_speed=85.4, loiter_speed=65.0,
        takeoff_altitude=457.2, landing_altitude=457.2,
    ),
    "c172": dict(
        mtom=1110, empty_mass=770, payload=180, span=11.0,
        flow_rate=0.0, can_scoop=False, scooping_distance=0,
        icon="plane.svg", takeoff_landing_type="runway",
        total_propellant=144, reserve_propellant=21.6, refueling_rate=0.5,
        taxi_out_fc=0.0016, taxi_in_fc=0.0016,
        takeoff_fc=0.0089,
        cruise_fc=0.007, cruise_climb_fc=0.007, cruise_descent_fc=0.003,
        landing_fc=0.001, loiter_fc=0.006,
        cruise_altitude=2438, cruise_speed=62.0, loiter_speed=50.0,
        takeoff_altitude=457.2, landing_altitude=457.2,
    ),
}


# ──────────────────────────────────────────────
# Spec-based model
# ──────────────────────────────────────────────

class SpecAircraftModel:
    """Spec-based aircraft model — direct pass-through, no sizing.

    Takes a PerformanceSpec (or a preset name) and outputs the dict
    expected by the wildfire simulation. No OpenMDAO, no mission
    analysis, no aerodynamic computation — just packaging.
    """

    def __init__(self, preset: str | None = None):
        """Initialize with an optional preset ('cl415', 'dhc515', 'at802f', 'c172')."""
        self.preset = preset

    def model(self, spec: PerformanceSpec | dict[str, Any] | None = None) -> dict[str, Any]:
        """Build the sim-compatible output from a performance spec.

        Args:
            spec: A PerformanceSpec dataclass, a dict of spec values, or
                  None to use the preset defaults.

        Returns:
            Dict matching the wildfire sim aircraft JSON format.
        """
        if spec is None:
            preset_name = self.preset or "cl415"
            cfg = SPECS.get(preset_name, SPECS["cl415"]).copy()
        elif isinstance(spec, PerformanceSpec):
            cfg = spec.__dict__.copy()
        elif isinstance(spec, dict):
            base = SPECS.get(self.preset or "cl415", SPECS["cl415"]).copy()
            base.update(spec)
            cfg = base
        else:
            raise TypeError(f"Expected PerformanceSpec or dict, got {type(spec)}")

        s = PerformanceSpec(**{k: v for k, v in cfg.items() if k in PerformanceSpec.__dataclass_fields__})

        propulsion_input = {
            "architecture": "conventional",
            "total_propellant": round(s.total_propellant, 1),
            "reserve_propellant": round(s.reserve_propellant, 1),
            "propellant_unit": "kg",
            "refueling_rate": s.refueling_rate,
            "taxi_out_fc": round(s.taxi_out_fc, 6),
            "taxi_in_fc": round(s.taxi_in_fc, 6),
            "takeoff_fc": round(s.takeoff_fc, 6),
            "transition_fc": round(s.transition_fc, 6),
            "retransition_fc": round(s.retransition_fc, 6),
            "cruise_fc": round(s.cruise_fc, 6),
            "cruise_climb_fc": round(s.cruise_climb_fc, 6),
            "cruise_descent_fc": round(s.cruise_descent_fc, 6),
            "landing_fc": round(s.landing_fc, 6),
            "loiter_fc": round(s.loiter_fc, 6),
        }

        profile_parameters = {
            "taxi_out_duration": s.taxi_out_duration,
            "taxi_in_duration": s.taxi_in_duration,
            "transition_duration": s.transition_duration,
            "retransition_duration": s.retransition_duration,
            "takeoff_altitude": s.takeoff_altitude,
            "takeoff_climb_rate": s.takeoff_climb_rate,
            "takeoff_ground_speed": s.takeoff_ground_speed,
            "cruise_altitude": s.cruise_altitude,
            "cruise_speed": s.cruise_speed,
            "cruise_climb_rate": s.cruise_climb_rate,
            "cruise_climb_ground_speed": s.cruise_climb_ground_speed,
            "cruise_descent_rate": s.cruise_descent_rate,
            "cruise_descent_ground_speed": s.cruise_descent_ground_speed,
            "landing_altitude": s.landing_altitude,
            "landing_descent_rate": s.landing_descent_rate,
            "landing_ground_speed": s.landing_ground_speed,
            "loiter_speed": s.loiter_speed,
        }

        sim_input = {
            "icon": s.icon,
            "takeoff_landing_type": s.takeoff_landing_type,
            "autonomous": s.autonomous,
            "mtom": round(s.mtom, 1),
            "empty_mass": round(s.empty_mass, 1),
            "payload": round(s.payload, 1),
            "flow_rate": s.flow_rate,
            "can_scoop": s.can_scoop,
            "scooping_distance": s.scooping_distance,
            "span": round(s.span, 2),
            "propulsion_input": propulsion_input,
            "profile_parameters": profile_parameters,
        }

        return sim_input

    def to_json(self, spec: PerformanceSpec | dict[str, Any] | None = None,
                indent: int = 2) -> str:
        """Build the sim output and return as a JSON string."""
        return json.dumps(self.model(spec), indent=indent)
