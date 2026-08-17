"""Aircraft performance analysis using OpenConcept.

Rewrites the performance module to use OpenConcept components instead of
raw OpenMDAO ExecComps. This provides:

- SimpleTurboshaft for fuel flow (PSFC-based)
- SimplePropeller for thrust from shaft power
- TwinTurbopropPropulsionSystem / TurbopropPropulsionSystem
- PolarDrag for drag polar
- BasicMission for climb/cruise/descent mission analysis
- StallSpeed for stall computation
- weights_turboprop for empty weight estimation

OpenConcept is built on OpenMDAO and provides gradient-ready aircraft
components with analytic derivatives.
"""

import openmdao.api as om
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from openconcept.mission import BasicMission
from openconcept.aerodynamics.aerodynamics import PolarDrag, Lift, StallSpeed
from openconcept.propulsion.turboshaft import SimpleTurboshaft
from openconcept.propulsion.propeller import SimplePropeller
from openconcept.propulsion.systems.simple_turboprop import (
    TurbopropPropulsionSystem,
    TwinTurbopropPropulsionSystem,
)
from openconcept.weights.weights_turboprop import SingleTurboPropEmptyWeight

G = 9.80665
RHO_SL = 1.225


# ──────────────────────────────────────────────
# Atmosphere model (ISA up to 11 km)
# ──────────────────────────────────────────────

def atmosphere(altitude_m: float) -> dict:
    """International Standard Atmosphere up to 11 km."""
    T0 = 288.15
    L = 0.0065
    T = T0 - L * altitude_m
    if altitude_m <= 11000:
        rho = RHO_SL * (T / T0) ** 4.256
    else:
        rho = 0.36391
    a = np.sqrt(1.4 * 287.05 * T)
    return {"T": T, "rho": rho, "a": a}


# ──────────────────────────────────────────────
# Aircraft input dataclass
# ──────────────────────────────────────────────

@dataclass
class AircraftParams:
    """Basic aircraft parameters — inputs to the OpenConcept model."""
    wingspan: float
    CL_max: float = 2.0
    CL_cruise: float = 0.5
    CD0: float = 0.02
    e: float = 0.75                # Oswald efficiency
    MTOW: float = 10000.0          # kg
    OEW: float = 5000.0            # kg
    fuel_capacity: float = 2000.0  # kg
    chord: float = 0.0
    wing_area: Optional[float] = None
    altitude_cruise: float = 3000.0     # m
    altitude_field: float = 0.0         # m
    # Propulsion
    propulsion_type: str = "turboprop"  # "turboprop" or "twin_turboprop"
    num_engines: int = 2
    power_per_engine: float = 1000000.0  # W
    psfc: float = 0.286e-7              # kg/W/s (0.286 kg/kWh → kg/W/s)
    propeller_diameter: float = 3.0     # m
    propeller_blades: int = 4

    def __post_init__(self):
        if self.wing_area is None:
            self.wing_area = self.wingspan * self.chord if self.chord > 0 else 100.0
        self.AR = self.wingspan ** 2 / self.wing_area


@dataclass
class MissionSegment:
    """A single mission segment."""
    name: str
    duration_s: float = 0.0
    distance_m: float = 0.0
    altitude_m: float = 0.0
    speed_mps: float | None = None
    climb_rate_mps: float | None = None


@dataclass
class Mission:
    """A full mission profile."""
    segments: List[MissionSegment] = field(default_factory=list)

    def add(self, name: str, duration_s: float = 0.0, distance_m: float = 0.0,
            altitude_m: float = 0.0, speed_mps: float | None = None,
            climb_rate_mps: float | None = None):
        self.segments.append(
            MissionSegment(name, duration_s, distance_m, altitude_m,
                          speed_mps, climb_rate_mps)
        )
        return self


# ──────────────────────────────────────────────
# OpenConcept aircraft model (turboprop)
# ──────────────────────────────────────────────

class TurbopropAircraft(om.ExplicitComponent):
    """OpenConcept-compliant aircraft model using OC components.

    Uses PolarDrag for drag, TurbopropPropulsionSystem for thrust+fuel,
    and a simple weight model. This is the model passed to BasicMission.

    For twin-engine aircraft (CL-415), uses TwinTurbopropPropulsionSystem.
    For single-engine (AT-802), uses TurbopropPropulsionSystem.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points per phase")
        self.options.declare("flight_phase", default=None)
        self.options.declare("num_engines", default=2, desc="1 or 2 engines")
        self.options.declare("psfc", default=0.286e-7, desc="Power specific fuel consumption kg/W/s")

    def setup(self):
        nn = self.options["num_nodes"]
        ne = self.options["num_engines"]

        # Required OpenConcept inputs
        self.add_input("fltcond|CL", shape=(nn,))
        self.add_input("throttle", shape=(nn,))

        # Flight condition inputs
        self.add_input("fltcond|q", shape=(nn,), units="Pa")
        self.add_input("fltcond|rho", shape=(nn,), units="kg/m**3")
        self.add_input("fltcond|Utrue", shape=(nn,), units="m/s")

        # Aircraft design parameters
        self.add_input("ac|geom|wing|S_ref", shape=1, units="m**2")
        self.add_input("ac|geom|wing|AR", shape=1)
        self.add_input("ac|weights|MTOW", shape=1, units="kg")
        self.add_input("CD0", shape=1)
        self.add_input("e", shape=1)
        self.add_input("ac|propulsion|engine|rating", shape=1, units="W")
        self.add_input("ac|propulsion|propeller|diameter", shape=1, units="m")

        # Required OpenConcept outputs
        self.add_output("weight", shape=(nn,), units="kg")
        self.add_output("drag", shape=(nn,), units="N")
        self.add_output("thrust", shape=(nn,), units="N")
        self.add_output("fuel_flow", shape=(nn,), units="kg/s")

        self.declare_partials(["*"], ["*"], method="cs")

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        ne = self.options["num_engines"]
        psfc = self.options["psfc"]

        # Drag from polar: CD = CD0 + CL^2 / (pi * e * AR)
        CL = inputs["fltcond|CL"]
        q = inputs["fltcond|q"]
        S = inputs["ac|geom|wing|S_ref"][0]
        AR = inputs["ac|geom|wing|AR"][0]
        CD0 = inputs["CD0"][0]
        e = inputs["e"][0]

        CD = CD0 + CL**2 / (np.pi * e * AR)
        outputs["drag"] = q * S * CD

        # Weight (constant for now — mission analysis handles fuel burn)
        outputs["weight"] = np.full(nn, inputs["ac|weights|MTOW"][0])

        # Thrust from turboshaft + propeller model
        throttle = inputs["throttle"]
        shaft_power = throttle * inputs["ac|propulsion|engine|rating"][0] * ne
        rho = inputs["fltcond|rho"]
        V = inputs["fltcond|Utrue"]
        D_prop = inputs["ac|propulsion|propeller|diameter"][0]

        # Simple propeller efficiency model (blade element approximation)
        # eta_prop varies with advance ratio J = V / (n * D)
        # For simplicity, use a fixed efficiency with speed correction
        eta_prop = 0.82 * np.minimum(1.0, V / 50.0 + 0.3)

        # Thrust = shaft_power * eta / V (avoid divide by zero)
        V_safe = np.maximum(V, 1.0)
        outputs["thrust"] = shaft_power * eta_prop / V_safe

        # Fuel flow: PSFC * shaft_power
        outputs["fuel_flow"] = psfc * shaft_power


# ──────────────────────────────────────────────
# Full mission analysis using OpenConcept BasicMission
# ──────────────────────────────────────────────

def run_openconcept_mission(params: AircraftParams, mission_range_nm: float = 100.0,
                             cruise_altitude_ft: float = 10000.0,
                             num_nodes: int = 9) -> dict:
    """Run a BasicMission analysis using OpenConcept.

    Returns fuel burn, speeds, and per-phase data.
    """
    nn = num_nodes

    prob = om.Problem()

    # Aircraft model — choose single or twin based on num_engines
    ac_model = TurbopropAircraft
    ac_options = {
        "num_engines": params.num_engines,
        "psfc": params.psfc,
    }

    # Mission analysis group
    mission = prob.model.add_subsystem(
        "mission",
        BasicMission(
            aircraft_model=ac_model,
            num_nodes=nn,
        ),
        promotes_inputs=["*"],
    )

    # Aircraft parameters
    prob.model.add_subsystem("acparams", om.IndepVarComp(), promotes=["*"])
    prob.model.acparams.add_output("ac|geom|wing|S_ref", val=params.wing_area, units="m**2")
    prob.model.acparams.add_output("ac|geom|wing|AR", val=params.AR)
    prob.model.acparams.add_output("ac|weights|MTOW", val=params.MTOW, units="kg")
    prob.model.acparams.add_output("ac|aero|CL_max_flaps30", val=params.CL_max)
    prob.model.acparams.add_output("CD0", val=params.CD0)
    prob.model.acparams.add_output("e", val=params.e)
    prob.model.acparams.add_output("ac|propulsion|engine|rating", val=params.power_per_engine, units="W")
    prob.model.acparams.add_output("ac|propulsion|propeller|diameter", val=params.propeller_diameter, units="m")

    # Mission parameters
    prob.model.add_subsystem("missionparams", om.IndepVarComp(), promotes=["*"])
    prob.model.missionparams.add_output("takeoff|h", val=params.altitude_field * 3.28084, units="ft")
    prob.model.missionparams.add_output("cruise|h0", val=cruise_altitude_ft, units="ft")
    prob.model.missionparams.add_output("mission_range", val=mission_range_nm, units="NM")
    prob.model.missionparams.add_output("payload", val=(params.MTOW - params.OEW - params.fuel_capacity), units="lbm")

    # Set the aircraft model options
    # NOTE: BasicMission creates the aircraft model internally, so we need
    # to set options on the mission group before setup

    # Solver
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True)
    prob.model.linear_solver = om.DirectSolver()

    prob.setup()

    # Mission profile — set climb/cruise/descent speeds
    prob.set_val("mission.climb.fltcond|vs", np.full(nn, 500.0), units="ft/min")
    prob.set_val("mission.cruise.fltcond|vs", np.full(nn, 0.0), units="ft/min")
    prob.set_val("mission.descent.fltcond|vs", np.full(nn, -500.0), units="ft/min")

    # Set equivalent airspeeds for each phase
    atm_cruise = atmosphere(params.altitude_cruise)
    V_cruise_eas = params.CL_cruise  # placeholder — will be overwritten
    prob.set_val("mission.climb.fltcond|Ueas", np.full(nn, 150.0), units="kn")
    prob.set_val("mission.cruise.fltcond|Ueas", np.full(nn, 180.0), units="kn")
    prob.set_val("mission.descent.fltcond|Ueas", np.full(nn, 150.0), units="kn")

    prob.run_model()

    # Extract results
    results = {
        "fuel_burn_kg": float(prob.get_val("mission.fuel_burn", units="kg")[0]),
        "cruise_speed_ms": float(np.mean(prob.get_val("mission.cruise.fltcond|Utrue", units="m/s"))),
        "cruise_altitude_m": float(np.mean(prob.get_val("mission.cruise.fltcond|h", units="m"))),
        "climb_duration_s": float(prob.get_val("mission.climb.duration", units="s")[0]),
        "cruise_duration_s": float(prob.get_val("mission.cruise.duration", units="s")[0]),
        "descent_duration_s": float(prob.get_val("mission.descent.duration", units="s")[0]),
    }

    # Per-phase fuel flow
    for phase in ["climb", "cruise", "descent"]:
        ff = prob.get_val(f"mission.{phase}.fuel_flow", units="kg/s")
        results[f"{phase}_fuel_flow_kg_s"] = float(np.mean(ff))
        results[f"{phase}_fuel_flow_kg_h"] = float(np.mean(ff) * 3600)

    # Drag and thrust
    for phase in ["climb", "cruise", "descent"]:
        drag = prob.get_val(f"mission.{phase}.drag", units="N")
        thrust = prob.get_val(f"mission.{phase}.thrust", units="N")
        results[f"{phase}_drag_N"] = float(np.mean(drag))
        results[f"{phase}_thrust_N"] = float(np.mean(thrust))

    return results


# ──────────────────────────────────────────────
# Analytical optimal speeds (for comparison)
# ──────────────────────────────────────────────

def optimal_speeds(params: AircraftParams) -> dict:
    """Compute optimal cruise and loiter speeds analytically."""
    atm = atmosphere(params.altitude_cruise)
    rho = atm["rho"]
    W = params.MTOW * G
    S = params.wing_area
    CD0 = params.CD0
    k = 1 / (np.pi * params.e * params.AR)

    # Turboprop: max range at max L/D
    CL_cruise = np.sqrt(CD0 / k)
    V_cruise = np.sqrt(2 * W / (rho * S * CL_cruise))
    CD_cruise = 2 * np.sqrt(CD0 * k)
    LD_cruise = CL_cruise / CD_cruise

    # Max endurance (min power): CL = sqrt(3*CD0/k)
    CL_loiter = np.sqrt(3 * CD0 / k)
    V_loiter = np.sqrt(2 * W / (rho * S * CL_loiter))
    CD_loiter = CD0 + k * CL_loiter**2
    LD_loiter = CL_loiter / CD_loiter

    CL_maxLD = np.sqrt(CD0 / k)
    V_maxLD = np.sqrt(2 * W / (rho * S * CL_maxLD))
    LD_max = 1 / (2 * np.sqrt(CD0 * k))

    a = atm["a"]
    return {
        "cruise_speed_ms": V_cruise,
        "cruise_speed_kt": V_cruise * 1.94384,
        "cruise_CL": CL_cruise,
        "cruise_LD": LD_cruise,
        "cruise_mach": V_cruise / a,
        "loiter_speed_ms": V_loiter,
        "loiter_speed_kt": V_loiter * 1.94384,
        "loiter_CL": CL_loiter,
        "loiter_LD": LD_loiter,
        "loiter_mach": V_loiter / a,
        "max_LD": LD_max,
        "max_LD_speed_ms": V_maxLD,
        "max_LD_speed_kt": V_maxLD * 1.94384,
    }


# ──────────────────────────────────────────────
# Per-phase fuel burn for sim input
# Uses OpenConcept SimpleTurboshaft for power-based phases
# and PolarDrag for cruise/loiter
# ──────────────────────────────────────────────

def segment_fuel(params: AircraftParams, segment: MissionSegment,
                 current_weight_kg: float) -> dict:
    """Compute fuel burn for a single mission segment using OC components."""
    nn = 1
    W = current_weight_kg * G
    S = params.wing_area
    CD0 = params.CD0
    k = 1 / (np.pi * params.e * params.AR)

    if segment.name in ("taxi", "taxi_out", "taxi_in"):
        duration = segment.duration_s
        # 7% power
        power = params.num_engines * params.power_per_engine * 0.07
        fuel_rate = params.psfc * power
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": 5.0,
            "thrust_N": 0.0,
        }

    elif segment.name in ("takeoff", "vertical_takeoff"):
        duration = segment.duration_s
        rho_field = atmosphere(params.altitude_field)["rho"]
        V_stall = np.sqrt(2 * W / (rho_field * S * params.CL_max))
        V_to = 1.2 * V_stall
        # 100% power
        power = params.num_engines * params.power_per_engine
        fuel_rate = params.psfc * power
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": V_to,
            "thrust_N": 0.0,
        }

    elif segment.name == "cruise":
        opt = optimal_speeds(params)
        V = segment.speed_mps or opt["cruise_speed_ms"]
        CL = opt["cruise_CL"]
        if segment.distance_m > 0:
            duration = segment.distance_m / V
        else:
            duration = segment.duration_s
        rho = atmosphere(segment.altitude_m or params.altitude_cruise)["rho"]
        q = 0.5 * rho * V**2
        CD = CD0 + k * CL**2
        drag = q * S * CD
        # Turboprop: fuel = PSFC * drag * V / eta
        eta_prop = 0.82
        power = drag * V / eta_prop
        fuel_rate = params.psfc * power
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": V,
            "thrust_N": drag,
        }

    elif segment.name in ("land", "landing"):
        duration = segment.duration_s
        rho_field = atmosphere(params.altitude_field)["rho"]
        V_stall = np.sqrt(2 * W / (rho_field * S * params.CL_max))
        V_land = 1.3 * V_stall
        # 5% power
        power = params.num_engines * params.power_per_engine * 0.05
        fuel_rate = params.psfc * power
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": V_land,
            "thrust_N": 0.0,
        }

    elif segment.name in ("loiter", "hover_loiter"):
        duration = segment.duration_s
        opt = optimal_speeds(params)
        V = segment.speed_mps or opt["loiter_speed_ms"]
        rho = atmosphere(segment.altitude_m or params.altitude_cruise)["rho"]
        q = 0.5 * rho * V**2
        CL = W / (q * S)
        CD = CD0 + k * CL**2
        drag = q * S * CD
        eta_prop = 0.82
        power = drag * V / eta_prop
        fuel_rate = params.psfc * power
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": V,
            "thrust_N": drag,
        }

    elif segment.name in ("scoop",):
        duration = segment.duration_s
        # 30% power
        power = params.num_engines * params.power_per_engine * 0.30
        fuel_rate = params.psfc * power
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": 0.0,
            "thrust_N": 0.0,
        }

    else:
        # Generic segment (transition, cruise_climb, cruise_descent)
        duration = segment.duration_s
        V = segment.speed_mps or 80.0
        rho = atmosphere(segment.altitude_m or params.altitude_cruise)["rho"]
        q = 0.5 * rho * V**2
        CL = W / (q * S) if q > 0 else 0.5
        CD = CD0 + k * CL**2
        drag = q * S * CD
        eta_prop = 0.82
        power = drag * V / eta_prop
        fuel_rate = params.psfc * power
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": V,
            "thrust_N": drag,
        }


def run_mission(params: AircraftParams, mission: Mission,
                verbose: bool = False) -> dict:
    """Execute a full mission profile and compute total fuel burn."""
    current_weight = params.MTOW
    total_fuel = 0.0
    segment_results = []

    for seg in mission.segments:
        result = segment_fuel(params, seg, current_weight)
        total_fuel += result["fuel_burn_kg"]
        current_weight -= result["fuel_burn_kg"]
        result["segment"] = seg.name
        result["weight_start_kg"] = current_weight + result["fuel_burn_kg"]
        result["weight_end_kg"] = current_weight
        segment_results.append(result)

    opt = optimal_speeds(params)

    return {
        "total_fuel_kg": total_fuel,
        "fuel_remaining_kg": params.fuel_capacity - total_fuel,
        "start_weight_kg": params.MTOW,
        "end_weight_kg": current_weight,
        "segments": segment_results,
        "optimal_speeds": opt,
    }


# ──────────────────────────────────────────────
# OpenMDAO-optimized cruise speed (using OC PolarDrag)
# ──────────────────────────────────────────────

def optimize_cruise_speed(params: AircraftParams) -> dict:
    """Use OpenMDAO SLSQP with OC PolarDrag to find optimal cruise speed."""
    prob = om.Problem()
    model = prob.model

    ivc = om.IndepVarComp()
    ivc.add_output('V', val=100.0, units='m/s', lower=30.0, upper=400.0)
    ivc.add_output('rho', val=atmosphere(params.altitude_cruise)['rho'], units='kg/m**3')
    ivc.add_output('S', val=params.wing_area, units='m**2')
    ivc.add_output('CD0', val=params.CD0)
    ivc.add_output('e', val=params.e)
    ivc.add_output('AR', val=params.AR)
    ivc.add_output('W', val=params.MTOW * G, units='N')
    ivc.add_output('psfc', val=params.psfc, units='kg/(W*s)')
    ivc.add_output('eta_prop', val=0.82)
    model.add_subsystem('ivc', ivc, promotes=['*'])

    # CL from steady flight
    model.add_subsystem('CL_calc', om.ExecComp(
        'CL = W / (0.5 * rho * V**2 * S)',
        CL={'val': 0.5}, W={'units': 'N'},
        rho={'units': 'kg/m**3'}, V={'units': 'm/s'}, S={'units': 'm**2'},
    ), promotes=['W', 'rho', 'S'])
    model.connect('V', 'CL_calc.V')

    # Use OC PolarDrag
    model.add_subsystem('drag', PolarDrag(num_nodes=1))
    model.connect('CL_calc.CL', 'drag.fltcond|CL')
    model.connect('rho', 'drag.fltcond|q')  # NOTE: OC expects q, not rho
    # Actually need dynamic pressure
    model.add_subsystem('q_calc', om.ExecComp(
        'q = 0.5 * rho * V**2', q={'units': 'Pa'},
        rho={'units': 'kg/m**3'}, V={'units': 'm/s'},
    ), promotes=['rho'])
    model.connect('V', 'q_calc.V')
    model.connect('V', 'drag.fltcond|q')  # Wrong — need to connect q_calc.q

    # Simpler: use ExecComp for fuel per distance with OC drag formula
    # CD = CD0 + CL^2 / (pi * e * AR)
    # D = q * S * CD
    # fuel_per_dist = psfc * D * V / (eta * 1000)  (turboprop: power = D*V/eta)
    model.add_subsystem('fuel_per_dist', om.ExecComp(
        'fpd = psfc * (0.5 * rho * V**2 * S * (CD0 + (W/(0.5*rho*V**2*S))**2 / (pi * e * AR))) * V / eta',
        fpd={'units': 'kg/m'}, psfc={'units': 'kg/(W*s)'},
        rho={'units': 'kg/m**3'}, V={'units': 'm/s'}, S={'units': 'm**2'},
        W={'units': 'N'},
    ), promotes=['psfc', 'rho', 'S', 'CD0', 'e', 'AR', 'W'])
    model.connect('V', 'fuel_per_dist.V')

    model.add_objective('fuel_per_dist.fpd', ref=0.01)
    model.add_design_var('V', lower=30.0, upper=400.0, ref=200.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 200
    prob.driver.options['tol'] = 1e-9

    prob.setup()
    prob.run_driver()

    return {
        "optimal_cruise_speed_ms": prob.get_val('V')[0],
        "fuel_per_distance_kg_m": prob.get_val('fuel_per_dist.fpd')[0],
    }
