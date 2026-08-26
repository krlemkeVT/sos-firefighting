"""Spec-based aircraft performance model — config in, performance out.

Takes configuration and dimension specs (wingspan, wing area, CD0, Oswald
efficiency, MTOW, power, PSFC, etc.) as inputs, computes performance via
OpenConcept/OpenMDAO (stall speed, optimal cruise/loiter speeds, max L/D,
per-segment fuel burn rates), and outputs a PerformanceSpec with the
computed values — plus a comparison against published reference data.

Unlike the sizer (which runs a full mission profile to compute fuel burn),
this model computes performance metrics directly from the aero model and
packages them. Use it when you want to validate the OpenConcept performance
output against known/published aircraft performance data.

Branch: feature/performance-spec-model
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from aircraft_sizing.performance import (
    AircraftParams,
    optimal_speeds,
    atmosphere,
    run_mission,
    Mission,
)


G = 9.80665


# ──────────────────────────────────────────────
# Configuration spec dataclass (INPUTS)
# ──────────────────────────────────────────────

@dataclass
class ConfigSpec:
    """Aircraft configuration and dimension specifications — inputs to the model.

    These are the physical/engineering parameters that define the aircraft.
    The model uses these to compute performance via OpenConcept/OpenMDAO.
    """

    # ── Geometry ──
    wingspan: float = 28.38              # m
    wing_area: float = 100.0             # m²
    cl_max: float = 2.19               # max lift coefficient
    cl_cruise: float = 0.43            # cruise lift coefficient
    cd0: float = 0.0414                # zero-lift drag coefficient
    e: float = 0.75                    # Oswald efficiency factor

    # ── Mass ──
    mtow: float = 19890                  # max takeoff weight (kg)
    oew: float = 12880                   # operating empty weight (kg)
    fuel_capacity: float = 4650         # fuel capacity (kg)

    # ── Propulsion ──
    propulsion_type: str = "turboprop"
    num_engines: int = 2
    power_per_engine: float = 1775000    # W
    psfc: float = 0.286 / 3.6e6          # kg/W/s (0.286 kg/kWh → kg/W/s)
    propeller_diameter: float = 3.97     # m
    propeller_blades: int = 4
    eta_prop: float = 0.82              # propeller efficiency

    # ── Operating conditions ──
    altitude_cruise: float = 3000.0      # m
    altitude_field: float = 457.2        # m (1,500 ft)

    # ── Firefighting config (pass-through) ──
    flow_rate: float = 1.2              # water drop flow rate
    can_scoop: bool = True
    scooping_distance: float = 410       # m
    icon: str = "seaplane.svg"
    takeoff_landing_type: str = "runway"
    refueling_rate: float = 7.7         # kg/s
    reserve_fraction: float = 0.15      # reserve fuel as fraction of total

    @property
    def AR(self) -> float:
        """Aspect ratio."""
        return self.wingspan ** 2 / self.wing_area

    @property
    def k(self) -> float:
        """Induced drag coefficient: k = 1/(pi * e * AR)."""
        return 1.0 / (math.pi * self.e * self.AR)

    def to_aircraft_params(self) -> AircraftParams:
        """Convert to AircraftParams for OpenConcept computation."""
        return AircraftParams(
            wingspan=self.wingspan,
            CL_max=self.cl_max,
            CL_cruise=self.cl_cruise,
            CD0=self.cd0,
            e=self.e,
            MTOW=self.mtow,
            OEW=self.oew,
            fuel_capacity=self.fuel_capacity,
            propulsion_type=self.propulsion_type,
            num_engines=self.num_engines,
            power_per_engine=self.power_per_engine,
            psfc=self.psfc,
            propeller_diameter=self.propeller_diameter,
            propeller_blades=self.propeller_blades,
            wing_area=self.wing_area,
            altitude_cruise=self.altitude_cruise,
            altitude_field=self.altitude_field,
        )


# ──────────────────────────────────────────────
# Performance spec dataclass (OUTPUTS)
# ──────────────────────────────────────────────

@dataclass
class PerformanceSpec:
    """Computed performance specifications — outputs from OpenConcept.

    All values are computed from ConfigSpec inputs via OpenConcept/OpenMDAO
    or analytical formulas. These are the performance numbers the wildfire
    simulation needs.
    """

    # ── Computed speeds ──
    stall_speed_ms: float = 0.0          # stall speed (m/s)
    stall_speed_kt: float = 0.0          # stall speed (kt)
    optimal_cruise_ms: float = 0.0       # optimal cruise speed (m/s)
    optimal_cruise_kt: float = 0.0       # optimal cruise speed (kt)
    optimal_loiter_ms: float = 0.0       # optimal loiter speed (m/s)
    optimal_loiter_kt: float = 0.0       # optimal loiter speed (kt)

    # ── Computed aerodynamics ──
    max_LD: float = 0.0                  # max lift-to-drag ratio
    cruise_LD: float = 0.0              # cruise L/D
    loiter_LD: float = 0.0             # loiter L/D
    cruise_cl: float = 0.0             # cruise lift coefficient
    loiter_cl: float = 0.0            # loiter lift coefficient
    cruise_mach: float = 0.0          # cruise Mach number

    # ── Computed fuel rates (kg/s) ──
    cruise_fc: float = 0.0             # cruise fuel consumption rate
    loiter_fc: float = 0.0             # loiter fuel consumption rate
    takeoff_fc: float = 0.0            # takeoff fuel consumption rate
    taxi_out_fc: float = 0.0           # taxi out fuel consumption rate
    taxi_in_fc: float = 0.0           # taxi in fuel consumption rate
    landing_fc: float = 0.0           # landing fuel consumption rate
    cruise_climb_fc: float = 0.0      # cruise climb fuel consumption rate
    cruise_descent_fc: float = 0.0   # cruise descent fuel consumption rate
    scoop_fc: float = 0.0            # scoop fuel consumption rate

    # ── Computed range/endurance ──
    ferry_range_km: float = 0.0         # ferry range at optimal cruise (km)
    max_endurance_h: float = 0.0        # max endurance at loiter (h)

    # ── Fuel ──
    total_fuel_kg: float = 0.0          # total fuel for mission (kg)
    fuel_remaining_kg: float = 0.0      # fuel remaining after mission (kg)


# ──────────────────────────────────────────────
# Published reference data for comparison
# ──────────────────────────────────────────────

PUBLISHED = {
    "cl415": {
        "stall_speed_kt": 68,
        "max_LD": 10.9,
        "lrc_fuel_kg_h": 597,
        "normal_cruise_fuel_kg_h": 597,  # DH publishes single average
        "ferry_range_km": 2333,
        "max_endurance_h": 6.3,
        "optimal_cruise_kt": 140,
        # Per-segment fuel rates from validate_aircraft.py (kg/s)
        "cruise_fc": 0.226,
        "loiter_fc": 0.232,
        "takeoff_fc": 0.495,
        "taxi_out_fc": 0.091,
        "landing_fc": 0.016,
        # Sources
        "_sources": {
            "stall_speed_kt": "De Havilland DHC-515 official",
            "max_LD": "Published glide ratio",
            "lrc_fuel_kg_h": "De Havilland (avg at LRC)",
            "ferry_range_km": "De Havilland (4,626 kg fuel)",
            "max_endurance_h": "Derived from loiter fuel burn",
            "optimal_cruise_kt": "De Havilland LRC 140 kt",
        },
    },
    "dhc515": {
        "stall_speed_kt": 68,
        "max_LD": 10.9,
        "lrc_fuel_kg_h": 597,
        "normal_cruise_fuel_kg_h": 597,
        "ferry_range_km": 2333,
        "max_endurance_h": 6.3,
        "optimal_cruise_kt": 140,
        "_sources": {
            "stall_speed_kt": "De Havilland DHC-515 official",
            "max_LD": "Published glide ratio",
            "lrc_fuel_kg_h": "De Havilland (avg at LRC)",
        },
    },
    "at802f": {
        "stall_speed_kt": 79,
        "max_LD": 12.5,
        "cruise_fuel_kg_h": 236,         # PlanePhD: 78 GPH at 166 kt
        "patrol_fuel_kg_h": 215,         # 802u.com: 71 GPH at 180 kt
        "optimal_cruise_kt": 166,
        "cruise_fc": 0.066,
        "loiter_fc": 0.046,
        "takeoff_fc": 0.101,
        "taxi_out_fc": 0.007,
        "landing_fc": 0.005,
        "_sources": {
            "stall_speed_kt": "Air Tractor / 802u.com",
            "max_LD": "Estimated",
            "cruise_fuel_kg_h": "PlanePhD: 78 GPH at 166 kt, 8000 ft",
            "patrol_fuel_kg_h": "802u.com: 71 GPH at 180 kt",
        },
    },
    "c172": {
        "stall_speed_kt": 40,
        "max_LD": 11.0,
        "optimal_cruise_kt": 122,
        "_sources": {
            "stall_speed_kt": "Cessna POH",
            "max_LD": "Cessna POH glide ratio",
            "optimal_cruise_kt": "Cessna POH",
        },
    },
}


# ──────────────────────────────────────────────
# Config presets — aircraft dimension/config inputs
# ──────────────────────────────────────────────

CONFIGS = {
    "cl415": dict(
        wingspan=28.38, wing_area=100.0, cl_max=2.19, cl_cruise=0.43,
        cd0=0.0414, e=0.75, mtow=19890, oew=12880, fuel_capacity=4650,
        propulsion_type="turboprop", num_engines=2, power_per_engine=1775000,
        psfc=0.286 / 3.6e6, propeller_diameter=3.97, propeller_blades=4,
        eta_prop=0.82, altitude_cruise=1500, altitude_field=457.2,
        flow_rate=1.2, can_scoop=True, scooping_distance=410,
        icon="seaplane.svg", takeoff_landing_type="runway",
        refueling_rate=7.7, reserve_fraction=0.15,
    ),
    "dhc515": dict(
        wingspan=28.6, wing_area=100.34, cl_max=2.19, cl_cruise=0.43,
        cd0=0.0414, e=0.75, mtow=20547, oew=12995, fuel_capacity=4626,
        propulsion_type="turboprop", num_engines=2, power_per_engine=1775000,
        psfc=0.286 / 3.6e6, propeller_diameter=3.97, propeller_blades=4,
        eta_prop=0.82, altitude_cruise=3000, altitude_field=457.2,
        flow_rate=1.2, can_scoop=True, scooping_distance=410,
        icon="seaplane.svg", takeoff_landing_type="runway",
        refueling_rate=7.7, reserve_fraction=0.15,
    ),
    "at802f": dict(
        wingspan=18.04, wing_area=37.25, cl_max=1.89, cl_cruise=0.70,
        cd0=0.019, e=0.75, mtow=7257, oew=3062, fuel_capacity=933,
        propulsion_type="turboprop", num_engines=1, power_per_engine=1010000,
        psfc=0.363 / 3.6e6, propeller_diameter=3.0, propeller_blades=5,
        eta_prop=0.80, altitude_cruise=2438, altitude_field=457.2,
        flow_rate=1.2, can_scoop=False, scooping_distance=0,
        icon="seaplane.svg", takeoff_landing_type="runway",
        refueling_rate=7.7, reserve_fraction=0.15,
    ),
    "c172": dict(
        wingspan=11.0, wing_area=16.5, cl_max=1.8, cl_cruise=0.40,
        cd0=0.028, e=0.70, mtow=1110, oew=770, fuel_capacity=144,
        propulsion_type="turboprop", num_engines=1, power_per_engine=120000,
        psfc=0.286 / 3.6e6, propeller_diameter=1.9, propeller_blades=2,
        eta_prop=0.80, altitude_cruise=2438, altitude_field=457.2,
        flow_rate=0.0, can_scoop=False, scooping_distance=0,
        icon="plane.svg", takeoff_landing_type="runway",
        refueling_rate=0.5, reserve_fraction=0.15,
    ),
}


# ──────────────────────────────────────────────
# Spec-based model: config → performance
# ──────────────────────────────────────────────

class SpecAircraftModel:
    """Spec-based aircraft performance model — config in, performance out.

    Takes a ConfigSpec (wingspan, wing area, CD0, e, MTOW, power, PSFC, etc.)
    and computes performance via OpenConcept/OpenMDAO: stall speed, optimal
    cruise/loiter speeds, max L/D, per-segment fuel burn rates, ferry range,
    and max endurance.

    Optionally compares computed performance against published reference data.
    """

    def __init__(self, preset: str | None = None):
        """Initialize with an optional preset ('cl415', 'dhc515', 'at802f', 'c172')."""
        self.preset = preset

    def _build_config(self, config: ConfigSpec | dict[str, Any] | None) -> ConfigSpec:
        """Build a ConfigSpec from a dataclass, dict, or preset name."""
        if config is None:
            name = self.preset or "cl415"
            cfg = CONFIGS.get(name, CONFIGS["cl415"])
        elif isinstance(config, ConfigSpec):
            return config
        elif isinstance(config, dict):
            base = CONFIGS.get(self.preset or "cl415", CONFIGS["cl415"]).copy()
            base.update(config)
            cfg = base
        else:
            raise TypeError(f"Expected ConfigSpec or dict, got {type(config)}")

        valid_fields = ConfigSpec.__dataclass_fields__
        return ConfigSpec(**{k: v for k, v in cfg.items() if k in valid_fields})

    def compute(self, config: ConfigSpec | dict[str, Any] | None = None) -> PerformanceSpec:
        """Compute performance from configuration specs.

        Args:
            config: ConfigSpec dataclass, dict of config values, or None for preset.

        Returns:
            PerformanceSpec with computed performance values.
        """
        cfg = self._build_config(config)
        params = cfg.to_aircraft_params()
        atm = atmosphere(params.altitude_cruise)
        rho = atm["rho"]
        a = atm["a"]
        W = params.MTOW * G
        S = params.wing_area
        k = 1 / (math.pi * params.e * params.AR)

        # ── Stall speed ──
        rho_field = atmosphere(params.altitude_field)["rho"]
        V_stall = math.sqrt(2 * W / (rho_field * S * params.CL_max))

        # ── Optimal speeds (analytical) ──
        opt = optimal_speeds(params)

        # ── Per-segment fuel rates ──
        # Run mission to get fuel rates for each phase
        mission = self._build_mission(
            cruise_altitude=params.altitude_cruise,
            cruise_speed=opt["cruise_speed_ms"],
            loiter_speed=opt["loiter_speed_ms"],
            altitude_field=params.altitude_field,
        )
        results = run_mission(params, mission, verbose=False)

        seg_rates = {}
        for seg in results["segments"]:
            seg_rates[seg["segment"]] = seg["fuel_rate_kg_s"]

        # ── Ferry range and endurance ──
        cruise_fuel_rate = seg_rates.get("cruise", 0.0)
        loiter_fuel_rate = seg_rates.get("loiter", 0.0)

        if cruise_fuel_rate > 0:
            ferry_range_km = (cfg.fuel_capacity / cruise_fuel_rate) * opt["cruise_speed_ms"] / 1000
        else:
            ferry_range_km = 0.0

        if loiter_fuel_rate > 0:
            max_endurance_h = (cfg.fuel_capacity / loiter_fuel_rate) / 3600
        else:
            max_endurance_h = 0.0

        return PerformanceSpec(
            # Speeds
            stall_speed_ms=V_stall,
            stall_speed_kt=V_stall * 1.94384,
            optimal_cruise_ms=opt["cruise_speed_ms"],
            optimal_cruise_kt=opt["cruise_speed_kt"],
            optimal_loiter_ms=opt["loiter_speed_ms"],
            optimal_loiter_kt=opt["loiter_speed_kt"],
            # Aerodynamics
            max_LD=opt["max_LD"],
            cruise_LD=opt["cruise_LD"],
            loiter_LD=opt["loiter_LD"],
            cruise_cl=opt["cruise_CL"],
            loiter_cl=opt["loiter_CL"],
            cruise_mach=opt["cruise_mach"],
            # Fuel rates (kg/s)
            cruise_fc=seg_rates.get("cruise", 0.0),
            loiter_fc=seg_rates.get("loiter", 0.0),
            takeoff_fc=seg_rates.get("takeoff", 0.0),
            taxi_out_fc=seg_rates.get("taxi_out", 0.0),
            taxi_in_fc=seg_rates.get("taxi_out", 0.0),
            landing_fc=seg_rates.get("landing", 0.0),
            cruise_climb_fc=seg_rates.get("cruise_climb", seg_rates.get("cruise", 0.0)),
            cruise_descent_fc=seg_rates.get("cruise_descent", 0.0),
            scoop_fc=seg_rates.get("scoop", 0.0),
            # Range/endurance
            ferry_range_km=ferry_range_km,
            max_endurance_h=max_endurance_h,
            # Fuel
            total_fuel_kg=results["total_fuel_kg"],
            fuel_remaining_kg=results["fuel_remaining_kg"],
        )

    def _build_mission(self, cruise_altitude: float = 3000.0,
                       cruise_speed: float | None = None,
                       loiter_speed: float | None = None,
                       altitude_field: float = 457.2) -> Mission:
        """Build a standard firefighting mission for fuel rate computation."""
        mission = Mission()
        mission.add("taxi_out", duration_s=120.0)
        mission.add("takeoff", duration_s=60.0, altitude_m=altitude_field,
                    climb_rate_mps=3.78)
        mission.add("cruise_climb", duration_s=120.0, altitude_m=cruise_altitude,
                    climb_rate_mps=8.85)
        mission.add("cruise", duration_s=1800.0,
                    altitude_m=cruise_altitude, speed_mps=cruise_speed)
        mission.add("cruise_descent", duration_s=120.0,
                    climb_rate_mps=-8.85)
        mission.add("loiter", duration_s=600.0,
                    speed_mps=loiter_speed)
        mission.add("landing", duration_s=60.0, altitude_m=altitude_field,
                    climb_rate_mps=-5.0)
        mission.add("scoop", duration_s=300.0)
        return mission

    def compare(self, config: ConfigSpec | dict[str, Any] | None = None,
                preset: str | None = None) -> dict[str, Any]:
        """Compute performance and compare against published reference data.

        Args:
            config: ConfigSpec or dict. If None, uses the preset.
            preset: Preset name for published data lookup. Defaults to self.preset.

        Returns:
            Dict with computed PerformanceSpec values and comparison to published.
        """
        perf = self.compute(config)
        preset_name = preset or self.preset or "cl415"
        published = PUBLISHED.get(preset_name, {})

        computed_vals = perf.__dict__.copy()
        comparisons = []
        for key, pub_val in published.items():
            if key.startswith("_"):
                continue
            comp_val = computed_vals.get(key, None)
            if comp_val is not None and pub_val != 0:
                pct = ((comp_val - pub_val) / pub_val) * 100
                comparisons.append({
                    "metric": key,
                    "computed": round(comp_val, 4),
                    "published": pub_val,
                    "pct_diff": round(pct, 1),
                    "source": published.get("_sources", {}).get(key, ""),
                })
            elif comp_val is not None and pub_val == 0:
                comparisons.append({
                    "metric": key,
                    "computed": round(comp_val, 4),
                    "published": pub_val,
                    "pct_diff": None,
                    "source": published.get("_sources", {}).get(key, ""),
                })

        return {
            "preset": preset_name,
            "performance": perf.__dict__,
            "comparison": comparisons,
        }

    def to_json(self, config: ConfigSpec | dict[str, Any] | None = None,
                indent: int = 2) -> str:
        """Compute performance and return as JSON string."""
        return json.dumps(self.compute(config).__dict__, indent=indent)
