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
import numpy as np
from dataclasses import dataclass, field
from typing import Any

import openmdao.api as om
from openconcept.mission import BasicMission

from aircraft_sizing.performance import (
    AircraftParams,
    optimal_speeds,
    atmosphere,
)


G = 9.80665


# ──────────────────────────────────────────────
# OpenConcept turboprop aircraft component
# ──────────────────────────────────────────────

def _make_turboprop_ac(num_engines: int, psfc: float):
    """Create an OpenConcept-compliant turboprop aircraft component class.

    The class uses a parabolic drag polar (CD0 + CL^2/(pi*e*AR)) and a
    PSFC-based fuel flow model (fuel = psfc * throttle * power * num_engines).
    Propeller efficiency varies with airspeed.
    """
    _ne = num_engines
    _psfc = psfc

    class TurbopropAC(om.ExplicitComponent):
        def initialize(self):
            self.options.declare("num_nodes", default=1)
            self.options.declare("flight_phase", default=None)

        def setup(self):
            nn = self.options["num_nodes"]
            self.add_input("fltcond|CL", shape=(nn,))
            self.add_input("throttle", shape=(nn,))
            self.add_input("fltcond|q", shape=(nn,), units="Pa")
            self.add_input("fltcond|rho", shape=(nn,), units="kg/m**3")
            self.add_input("fltcond|Utrue", shape=(nn,), units="m/s")
            self.add_input("ac|geom|wing|S_ref", shape=1, units="m**2")
            self.add_input("ac|geom|wing|AR", shape=1)
            self.add_input("ac|weights|MTOW", shape=1, units="kg")
            self.add_input("CD0", shape=1)
            self.add_input("e", shape=1)
            self.add_input("ac|propulsion|engine|rating", shape=1, units="W")
            self.add_input("ac|propulsion|propeller|diameter", shape=1, units="m")
            self.add_output("weight", shape=(nn,), units="kg")
            self.add_output("drag", shape=(nn,), units="N")
            self.add_output("thrust", shape=(nn,), units="N")
            self.add_output("fuel_flow", shape=(nn,), units="kg/s")
            self.declare_partials(["*"], ["*"], method="cs")

        def compute(self, inputs, outputs):
            nn = self.options["num_nodes"]
            CL = inputs["fltcond|CL"]
            q = inputs["fltcond|q"]
            S = inputs["ac|geom|wing|S_ref"][0]
            AR = inputs["ac|geom|wing|AR"][0]
            CD0 = inputs["CD0"][0]
            e_val = inputs["e"][0]
            CD = CD0 + CL**2 / (np.pi * e_val * AR)
            outputs["drag"] = q * S * CD
            outputs["weight"] = np.full(nn, inputs["ac|weights|MTOW"][0])
            throttle = inputs["throttle"]
            shaft_power = throttle * inputs["ac|propulsion|engine|rating"][0] * _ne
            V = inputs["fltcond|Utrue"]
            eta_prop = 0.82 * np.minimum(1.0, V / 50.0 + 0.3)
            V_safe = np.maximum(V, 1.0)
            outputs["thrust"] = shaft_power * eta_prop / V_safe
            outputs["fuel_flow"] = _psfc * shaft_power

    return TurbopropAC


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
        """Compute performance from configuration specs via OpenConcept.

        Uses OpenConcept's BasicMission (climb, cruise, descent) to compute
        per-phase fuel flow, drag, thrust, CL, and true airspeed through
        OpenMDAO's atmospheric model and steady-flight solver. Stall speed,
        optimal speeds, and max L/D use analytical formulas (standard
        aerodynamic relationships that OpenConcept doesn't expose directly).

        Power-setting phases (taxi, takeoff, landing, scoop) use the same
        PSFC × power_fraction model as the analytical path.

        Args:
            config: ConfigSpec dataclass, dict of config values, or None for preset.

        Returns:
            PerformanceSpec with computed performance values.
        """
        cfg = self._build_config(config)
        params = cfg.to_aircraft_params()
        atm = atmosphere(params.altitude_cruise)
        W = params.MTOW * G
        S = params.wing_area

        # ── Stall speed (analytical) ──
        rho_field = atmosphere(params.altitude_field)["rho"]
        V_stall = math.sqrt(2 * W / (rho_field * S * params.CL_max))

        # ── Optimal speeds and max L/D (analytical) ──
        opt = optimal_speeds(params)

        # ── OpenConcept BasicMission (climb, cruise, descent) ──
        oc_results = self._run_openconcept(cfg, params)

        # Extract per-phase data from OpenConcept
        oc_phases = {}
        for phase in ["climb", "cruise", "descent"]:
            oc_phases[phase] = {
                "fuel_flow": oc_results[phase]["fuel_flow"],
                "drag": oc_results[phase]["drag"],
                "thrust": oc_results[phase]["thrust"],
                "duration": oc_results[phase]["duration"],
                "Utrue": oc_results[phase]["Utrue"],
                "CL": oc_results[phase]["CL"],
                "fuel_burn": oc_results[phase]["fuel_burn"],
            }

        # ── Power-setting phases (PSFC × power_fraction) ──
        # These are the same formulas used in the analytical model
        power_total = params.num_engines * params.power_per_engine

        taxi_rate = params.psfc * power_total * 0.07
        takeoff_rate = params.psfc * power_total * 1.0
        landing_rate = params.psfc * power_total * 0.05
        scoop_rate = params.psfc * power_total * 0.30

        # ── Cruise and loiter fuel rates from OpenConcept ──
        cruise_rate = oc_phases["cruise"]["fuel_flow"]
        cruise_climb_rate = oc_phases["climb"]["fuel_flow"]
        cruise_descent_rate = oc_phases["descent"]["fuel_flow"]

        # Loiter: use the analytical loiter speed at cruise altitude
        # (OpenConcept BasicMission doesn't have a loiter phase)
        loiter_rate = self._compute_loiter_fuel(cfg, params, opt["loiter_speed_ms"])

        # ── Total fuel from OpenConcept mission ──
        oc_total_fuel = sum(p["fuel_burn"] for p in oc_phases.values())

        # Add power-setting phase fuel
        ps_fuel = (
            taxi_rate * 120 +       # taxi_out
            takeoff_rate * 60 +    # takeoff
            landing_rate * 60 +    # landing
            scoop_rate * 300        # scoop
        )
        total_fuel = oc_total_fuel + ps_fuel + loiter_rate * 600

        # ── Ferry range and endurance ──
        if cruise_rate > 0:
            ferry_range_km = (cfg.fuel_capacity / cruise_rate) * opt["cruise_speed_ms"] / 1000
        else:
            ferry_range_km = 0.0

        if loiter_rate > 0:
            max_endurance_h = (cfg.fuel_capacity / loiter_rate) / 3600
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
            # Fuel rates (kg/s) — cruise/climb/descent from OpenConcept
            cruise_fc=cruise_rate,
            loiter_fc=loiter_rate,
            takeoff_fc=takeoff_rate,
            taxi_out_fc=taxi_rate,
            taxi_in_fc=taxi_rate,
            landing_fc=landing_rate,
            cruise_climb_fc=cruise_climb_rate,
            cruise_descent_fc=cruise_descent_rate,
            scoop_fc=scoop_rate,
            # Range/endurance
            ferry_range_km=ferry_range_km,
            max_endurance_h=max_endurance_h,
            # Fuel
            total_fuel_kg=total_fuel,
            fuel_remaining_kg=cfg.fuel_capacity - total_fuel,
        )

    def _run_openconcept(self, cfg: ConfigSpec, params: AircraftParams) -> dict:
        """Run OpenConcept BasicMission and extract per-phase results.

        Returns dict with climb/cruise/descent keys, each containing:
        fuel_flow (kg/s), drag (N), thrust (N), duration (s),
        Utrue (m/s), CL, fuel_burn (kg).
        """
        nn = 5
        ACClass = _make_turboprop_ac(cfg.num_engines, cfg.psfc)

        prob = om.Problem()
        prob.model.add_subsystem(
            "mission",
            BasicMission(aircraft_model=ACClass, num_nodes=nn),
            promotes_inputs=["*"],
        )

        # Aircraft parameters
        ivc = prob.model.add_subsystem("acparams", om.IndepVarComp(), promotes=["*"])
        ivc.add_output("ac|geom|wing|S_ref", val=cfg.wing_area, units="m**2")
        ivc.add_output("ac|geom|wing|AR", val=cfg.AR)
        ivc.add_output("ac|weights|MTOW", val=cfg.mtow, units="kg")
        ivc.add_output("ac|aero|CL_max_flaps30", val=cfg.cl_max)
        ivc.add_output("CD0", val=cfg.cd0)
        ivc.add_output("e", val=cfg.e)
        ivc.add_output("ac|propulsion|engine|rating", val=cfg.power_per_engine, units="W")
        ivc.add_output("ac|propulsion|propeller|diameter", val=cfg.propeller_diameter, units="m")

        # Mission parameters
        ivc2 = prob.model.add_subsystem("missionparams", om.IndepVarComp(), promotes=["*"])
        ivc2.add_output("takeoff|h", val=cfg.altitude_field * 3.28084, units="ft")
        ivc2.add_output("cruise|h0", val=cfg.altitude_cruise * 3.28084, units="ft")
        ivc2.add_output("mission_range", val=100.0, units="NM")
        ivc2.add_output("payload", val=(cfg.mtow - cfg.oew - cfg.fuel_capacity) * 2.20462, units="lbm")

        prob.setup()

        # Set equivalent airspeeds for each phase
        prob.set_val("mission.climb.atmos.trueair.fltcond|Ueas", np.full(nn, 150.0), units="kn")
        prob.set_val("mission.cruise.atmos.trueair.fltcond|Ueas", np.full(nn, 180.0), units="kn")
        prob.set_val("mission.descent.atmos.trueair.fltcond|Ueas", np.full(nn, 150.0), units="kn")

        prob.run_model()

        results = {}
        for phase in ["climb", "cruise", "descent"]:
            ff = prob.get_val(f"mission.{phase}.acmodel.fuel_flow", units="kg/s")
            dur = prob.get_val(f"mission.{phase}.ode_integ_phase.duration", units="s")
            drag = prob.get_val(f"mission.{phase}.acmodel.drag", units="N")
            thrust = prob.get_val(f"mission.{phase}.acmodel.thrust", units="N")
            Utrue = prob.get_val(f"mission.{phase}.fltcond|Utrue", units="m/s")
            CL = prob.get_val(f"mission.{phase}.fltcond|CL")
            results[phase] = {
                "fuel_flow": float(np.mean(ff)),
                "duration": float(dur[0]),
                "drag": float(np.mean(drag)),
                "thrust": float(np.mean(thrust)),
                "Utrue": float(np.mean(Utrue)),
                "CL": float(np.mean(CL)),
                "fuel_burn": float(np.mean(ff) * dur[0]),
            }
        return results

    def _compute_loiter_fuel(self, cfg: ConfigSpec, params: AircraftParams,
                             loiter_speed: float) -> float:
        """Compute loiter fuel rate at loiter speed and cruise altitude.

        Uses the same drag polar as the OpenConcept component: CD = CD0 + CL^2/(pi*e*AR).
        Fuel = psfc * drag * V / eta_prop.
        """
        atm = atmosphere(params.altitude_cruise)
        rho = atm["rho"]
        W = params.MTOW * G
        k = 1 / (math.pi * params.e * params.AR)
        V = loiter_speed
        q = 0.5 * rho * V**2
        CL = W / (q * params.wing_area)
        CD = params.CD0 + k * CL**2
        drag = q * params.wing_area * CD
        power = drag * V / cfg.eta_prop
        return params.psfc * power

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
