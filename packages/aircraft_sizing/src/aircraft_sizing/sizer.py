"""Aircraft sizer bridging OpenConcept performance to the TRITON sim.

Takes DesignVariables (from triton_api), builds an AircraftParams, runs
the OpenConcept/OpenMDAO performance model, and outputs a SimulationInput-
compatible dict matching the wildfire simulation's aircraft JSON format.
"""

from __future__ import annotations

import json
import math
from typing import Any

from aircraft_sizing.performance import (
    AircraftParams,
    Mission,
    run_mission,
    optimal_speeds,
    atmosphere,
)


# ──────────────────────────────────────────────
# Presets — aircraft parameter sets
# Uses Oswald efficiency e instead of k (k = 1/(pi*e*AR))
# Uses psfc (kg/W/s) instead of bsfc (kg/kWh) — psfc = bsfc/3.6e6
# ──────────────────────────────────────────────

PRESETS = {
    "cl415": dict(
        wingspan=28.38, wing_area=100.0, cl_max=2.19, cl_cruise=0.43,
        cd0=0.0414, e=0.75, mtow=19890, oew=12880, fuel=4650,
        psfc=0.286/3.6e6, power=1775000, engines=2,
        propeller_diameter=3.97, propeller_blades=4,
        altitude=1500, propulsion="turboprop",
        icon="seaplane.svg", takeoff_landing_type="runway",
        flow_rate=1.2, can_scoop=True, scooping_distance=410,
    ),
    "dhc515": dict(
        wingspan=28.6, wing_area=100.34, cl_max=2.19, cl_cruise=0.43,
        cd0=0.0414, e=0.75, mtow=20547, oew=12995, fuel=4626,
        psfc=0.286/3.6e6, power=1775000, engines=2,
        propeller_diameter=3.97, propeller_blades=4,
        altitude=3000, propulsion="turboprop",
        icon="seaplane.svg", takeoff_landing_type="runway",
        flow_rate=1.2, can_scoop=True, scooping_distance=410,
    ),
    "at802f": dict(
        wingspan=18.04, wing_area=37.25, cl_max=1.89, cl_cruise=0.7,
        cd0=0.019, e=0.75, mtow=7257, oew=3062, fuel=933,
        psfc=0.363/3.6e6, power=1010000, engines=1,
        propeller_diameter=3.0, propeller_blades=5,
        altitude=2438, propulsion="turboprop",
        icon="seaplane.svg", takeoff_landing_type="runway",
        flow_rate=1.2, can_scoop=False, scooping_distance=0,
    ),
    "c172": dict(
        wingspan=11.0, wing_area=16.5, cl_max=1.8, cl_cruise=0.4,
        cd0=0.028, e=0.7, mtow=1110, oew=770, fuel=144,
        psfc=0.286/3.6e6, power=120000, engines=1,
        propeller_diameter=1.9, propeller_blades=2,
        altitude=2438, propulsion="turboprop",
        icon="plane.svg", takeoff_landing_type="runway",
        flow_rate=0.0, can_scoop=False, scooping_distance=0,
    ),
}


# ──────────────────────────────────────────────
# Firefighting mission builder
# ──────────────────────────────────────────────

def build_firefighting_mission(
    cruise_altitude: float = 3000.0,
    cruise_speed: float | None = None,
    loiter_speed: float | None = None,
    cruise_duration: float = 1800.0,
    loiter_duration: float = 600.0,
) -> Mission:
    """Build a firefighting mission profile.

    Segments: taxi_out, takeoff, cruise_climb, cruise, cruise_descent,
    loiter, landing, scoop.
    """
    mission = Mission()
    mission.add("taxi_out", duration_s=120.0)
    mission.add("takeoff", duration_s=60.0, altitude_m=457.2,
                climb_rate_mps=3.78)
    mission.add("cruise_climb", duration_s=120.0, altitude_m=cruise_altitude,
                climb_rate_mps=8.85)
    mission.add("cruise", duration_s=cruise_duration,
                altitude_m=cruise_altitude, speed_mps=cruise_speed)
    mission.add("cruise_descent", duration_s=120.0,
                climb_rate_mps=-8.85)
    mission.add("loiter", duration_s=loiter_duration,
                speed_mps=loiter_speed)
    mission.add("landing", duration_s=60.0, altitude_m=457.2,
                climb_rate_mps=-5.0)
    mission.add("scoop", duration_s=300.0)
    return mission


# ──────────────────────────────────────────────
# Default sizer
# ──────────────────────────────────────────────

class DefaultAircraftSizer:
    """Aircraft sizer using OpenConcept performance models.

    Takes basic design variables, runs the performance analysis,
    and outputs a dict in the wildfire simulation's aircraft JSON format.
    """

    def __init__(self, preset: str | None = None):
        """Initialize with an optional preset ('cl415', 'dhc515', 'at802f', 'c172')."""
        self.preset = preset

    def size(self, design: dict[str, float] | None = None) -> dict[str, Any]:
        """Size the aircraft and return sim-compatible output."""
        cfg = PRESETS.get(self.preset, PRESETS["cl415"]).copy()

        if design:
            if "wing_area_m2" in design:
                cfg["wing_area"] = design["wing_area_m2"]
            if "aspect_ratio" in design:
                span_sq = design["aspect_ratio"] * cfg.get("wing_area", 100.0)
                cfg["wingspan"] = math.sqrt(span_sq)
            if "payload_kg" in design:
                cfg["payload"] = design["payload_kg"]
            if "cruise_speed_mps" in design:
                cfg["cruise_speed"] = design["cruise_speed_mps"]
            if "fuel_mass_kg" in design:
                cfg["fuel"] = design["fuel_mass_kg"]
            if "max_takeoff_mass_kg" in design:
                cfg["mtow"] = design["max_takeoff_mass_kg"]

        params = AircraftParams(
            wingspan=cfg["wingspan"],
            CL_max=cfg["cl_max"],
            CL_cruise=cfg["cl_cruise"],
            CD0=cfg["cd0"],
            e=cfg.get("e", 0.75),
            MTOW=cfg["mtow"],
            OEW=cfg["oew"],
            fuel_capacity=cfg["fuel"],
            propulsion_type=cfg.get("propulsion", "turboprop"),
            num_engines=cfg.get("engines", 2),
            power_per_engine=cfg.get("power", 1000000.0),
            psfc=cfg.get("psfc", 0.286e-7),
            propeller_diameter=cfg.get("propeller_diameter", 3.0),
            propeller_blades=cfg.get("propeller_blades", 4),
            wing_area=cfg.get("wing_area"),
            altitude_cruise=cfg.get("altitude", 3000.0),
            altitude_field=457.2,
        )

        opt = optimal_speeds(params)
        cruise_speed = cfg.get("cruise_speed", opt["cruise_speed_ms"])
        loiter_speed = opt["loiter_speed_ms"]

        mission = build_firefighting_mission(
            cruise_altitude=params.altitude_cruise,
            cruise_speed=cruise_speed,
            loiter_speed=loiter_speed,
        )
        results = run_mission(params, mission, verbose=False)

        seg_rates = {}
        seg_durations = {}
        for seg in results["segments"]:
            name = seg["segment"]
            fc_key = f"{name}_fc"
            seg_rates[fc_key] = round(seg["fuel_rate_kg_s"], 6)
            seg_durations[name] = seg["duration_s"]

        propulsion_input = {
            "architecture": "conventional",
            "total_propellant": round(cfg["fuel"], 1),
            "reserve_propellant": round(0.15 * cfg["fuel"], 1),
            "propellant_unit": "kg",
            "refueling_rate": 7.7,
            "taxi_out_fc": seg_rates.get("taxi_out_fc", 0.091),
            "taxi_in_fc": seg_rates.get("taxi_out_fc", 0.091),
            "takeoff_fc": seg_rates.get("takeoff_fc", 0.495),
            "transition_fc": 0.0,
            "retransition_fc": 0.0,
            "cruise_fc": seg_rates.get("cruise_fc", 0.226),
            "cruise_climb_fc": seg_rates.get("cruise_climb_fc",
                                             seg_rates.get("cruise_fc", 0.41)),
            "cruise_descent_fc": seg_rates.get("cruise_descent_fc", 0.052),
            "landing_fc": seg_rates.get("landing_fc", 0.016),
            "loiter_fc": seg_rates.get("loiter_fc", 0.232),
        }

        profile_parameters = {
            "taxi_out_duration": seg_durations.get("taxi_out", 120.0),
            "taxi_in_duration": 120.0,
            "transition_duration": 0.0,
            "retransition_duration": 0.0,
            "takeoff_altitude": 457.2,
            "takeoff_climb_rate": 3.78,
            "takeoff_ground_speed": 18.25,
            "cruise_altitude": params.altitude_cruise,
            "cruise_speed": cruise_speed,
            "cruise_climb_rate": 8.85,
            "cruise_climb_ground_speed": 84.55,
            "cruise_descent_rate": 8.85,
            "cruise_descent_ground_speed": 84.55,
            "landing_altitude": 457.2,
            "landing_descent_rate": 5.0,
            "landing_ground_speed": 18.25,
            "loiter_speed": loiter_speed,
        }

        sim_input = {
            "icon": cfg.get("icon", "seaplane.svg"),
            "takeoff_landing_type": cfg.get("takeoff_landing_type", "runway"),
            "autonomous": False,
            "mtom": round(params.MTOW, 1),
            "empty_mass": round(params.OEW, 1),
            "payload": round(cfg.get("payload", 6200.0), 1),
            "flow_rate": cfg.get("flow_rate", 1.2),
            "can_scoop": cfg.get("can_scoop", True),
            "scooping_distance": cfg.get("scooping_distance", 410),
            "span": round(params.wingspan, 2),
            "propulsion_input": propulsion_input,
            "profile_parameters": profile_parameters,
        }

        sim_input["_performance"] = {
            "total_fuel_kg": round(results["total_fuel_kg"], 1),
            "optimal_cruise_ms": round(opt["cruise_speed_ms"], 1),
            "optimal_cruise_kt": round(opt["cruise_speed_kt"], 0),
            "optimal_loiter_ms": round(opt["loiter_speed_ms"], 1),
            "optimal_loiter_kt": round(opt["loiter_speed_kt"], 0),
            "max_LD": round(opt["max_LD"], 1),
            "cruise_mach": round(opt["cruise_mach"], 3),
        }

        return sim_input

    def to_json(self, design: dict[str, float] | None = None,
                indent: int = 2) -> str:
        """Size the aircraft and return a JSON string."""
        return json.dumps(self.size(design), indent=indent)
