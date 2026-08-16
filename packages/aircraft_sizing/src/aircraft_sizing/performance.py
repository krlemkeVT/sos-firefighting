"""Aircraft performance analysis using OpenMDAO.

Computes fuel burn per mission segment (taxi, takeoff, cruise, land),
optimal cruise speed, and optimal loiter speed from basic aircraft stats.

Supports both jet (TSFC) and turboprop (BSFC + propeller efficiency)
propulsion types.
"""

import openmdao.api as om
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

G = 9.80665            # m/s²
RHO_SL = 1.225         # kg/m³ at sea level
RHO_11KM = 0.36391     # kg/m³ at 11 km (tropopause)


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
        rho = RHO_11KM
    a = np.sqrt(1.4 * 287.05 * T)
    return {"T": T, "rho": rho, "a": a}


# ──────────────────────────────────────────────
# Aircraft input dataclass
# ──────────────────────────────────────────────

@dataclass
class AircraftParams:
    """Basic aircraft parameters — the inputs to the performance model."""
    wingspan: float
    CL_max: float = 2.0
    CL_cruise: float = 0.5
    CD0: float = 0.02
    k: float = 0.05
    MTOW: float = 10000.0     # kg
    OEW: float = 5000.0       # kg
    fuel_capacity: float = 2000.0  # kg
    chord: float = 0.0
    TSFC: float = 1.78e-5    # 1/s (jet)
    num_engines: int = 2
    thrust_per_engine: float = 50000.0  # N
    wing_area: Optional[float] = None
    altitude_cruise: float = 10668.0    # m
    altitude_field: float = 0.0        # m
    propulsion_type: str = "jet"       # "jet" or "turboprop"
    BSFC: float = 0.0                  # kg/(kW·s)
    eta_prop: float = 0.82
    power_per_engine: float = 0.0      # W

    def __post_init__(self):
        if self.wing_area is None:
            self.wing_area = self.wingspan * self.chord
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
# Analytical optimal speeds
# ──────────────────────────────────────────────

def optimal_speeds(params: AircraftParams) -> dict:
    """Compute optimal cruise and loiter speeds analytically."""
    atm = atmosphere(params.altitude_cruise)
    rho = atm['rho']
    W = params.MTOW * G
    S = params.wing_area
    CD0 = params.CD0
    k = params.k

    if params.propulsion_type == "turboprop":
        CL_cruise = np.sqrt(CD0 / k)
        V_cruise = np.sqrt(2 * W / (rho * S * CL_cruise))
        CD_cruise = 2 * np.sqrt(CD0 * k)
        LD_cruise = CL_cruise / CD_cruise

        CL_loiter = np.sqrt(3 * CD0 / k)
        V_loiter = np.sqrt(2 * W / (rho * S * CL_loiter))
        CD_loiter = CD0 + k * CL_loiter ** 2
        LD_loiter = CL_loiter / CD_loiter
    else:
        CL_cruise = np.sqrt(CD0 / (3 * k))
        V_cruise = np.sqrt(2 * W / (rho * S * CL_cruise))
        CD_cruise = CD0 + k * CL_cruise ** 2
        LD_cruise = CL_cruise / CD_cruise

        CL_loiter = np.sqrt(CD0 / k)
        V_loiter = np.sqrt(2 * W / (rho * S * CL_loiter))
        CD_loiter = 2 * np.sqrt(CD0 * k)
        LD_loiter = CL_loiter / CD_loiter

    CL_maxLD = np.sqrt(CD0 / k)
    V_maxLD = np.sqrt(2 * W / (rho * S * CL_maxLD))
    LD_max = 1 / (2 * np.sqrt(CD0 * k))

    a = atm['a']
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
# Fuel rate helper
# ──────────────────────────────────────────────

def _fuel_rate(params: AircraftParams, drag_N: float, V_ms: float) -> float:
    """Compute fuel burn rate (kg/s) for the propulsion type."""
    if params.propulsion_type == "turboprop":
        power_W = drag_N * V_ms / params.eta_prop
        return params.BSFC * power_W / 1000.0
    else:
        return params.TSFC * drag_N


# ──────────────────────────────────────────────
# Mission fuel burn calculation
# ──────────────────────────────────────────────

def segment_fuel(params: AircraftParams, segment: MissionSegment,
                 current_weight_kg: float) -> dict:
    """Compute fuel burn for a single mission segment."""
    W = current_weight_kg * G
    S = params.wing_area
    CD0 = params.CD0
    k = params.k

    if segment.name in ("taxi", "taxi_out", "taxi_in"):
        duration = segment.duration_s
        if params.propulsion_type == "turboprop":
            power = params.num_engines * params.power_per_engine * 0.07
            fuel_rate = params.BSFC * power / 1000.0
            thrust = power * params.eta_prop / 5.0
        else:
            thrust = params.num_engines * params.thrust_per_engine * 0.07
            fuel_rate = params.TSFC * thrust
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": 5.0,
            "thrust_N": thrust,
        }

    elif segment.name in ("takeoff", "vertical_takeoff"):
        duration = segment.duration_s
        rho_field = atmosphere(params.altitude_field)["rho"]
        V_stall = np.sqrt(2 * W / (rho_field * S * params.CL_max))
        V_to = 1.2 * V_stall
        if params.propulsion_type == "turboprop":
            power = params.num_engines * params.power_per_engine
            fuel_rate = params.BSFC * power / 1000.0
            thrust = power * params.eta_prop / V_to
        else:
            thrust = params.num_engines * params.thrust_per_engine
            fuel_rate = params.TSFC * thrust
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": V_to,
            "thrust_N": thrust,
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
        q = 0.5 * rho * V ** 2
        CD = CD0 + k * CL ** 2
        drag = q * S * CD
        fuel_rate = _fuel_rate(params, drag, V)
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
        if params.propulsion_type == "turboprop":
            power = params.num_engines * params.power_per_engine * 0.05
            fuel_rate = params.BSFC * power / 1000.0
            thrust = power * params.eta_prop / V_land
        else:
            thrust = params.num_engines * params.thrust_per_engine * 0.05
            fuel_rate = params.TSFC * thrust
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": V_land,
            "thrust_N": thrust,
        }

    elif segment.name in ("loiter", "hover_loiter"):
        duration = segment.duration_s
        opt = optimal_speeds(params)
        V = segment.speed_mps or opt["loiter_speed_ms"]
        rho = atmosphere(segment.altitude_m or params.altitude_cruise)["rho"]
        q = 0.5 * rho * V ** 2
        CL = W / (q * S)
        CD = CD0 + k * CL ** 2
        drag = q * S * CD
        fuel_rate = _fuel_rate(params, drag, V)
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": V,
            "thrust_N": drag,
        }

    elif segment.name in ("scoop",):
        duration = segment.duration_s
        if params.propulsion_type == "turboprop":
            power = params.num_engines * params.power_per_engine * 0.30
            fuel_rate = params.BSFC * power / 1000.0
        else:
            thrust = params.num_engines * params.thrust_per_engine * 0.30
            fuel_rate = params.TSFC * thrust
        return {
            "fuel_burn_kg": fuel_rate * duration,
            "fuel_rate_kg_s": fuel_rate,
            "duration_s": duration,
            "speed_ms": 0.0,
            "thrust_N": 0.0,
        }

    else:
        # Generic segment (transition, cruise_climb, cruise_descent, etc.)
        duration = segment.duration_s
        V = segment.speed_mps or 80.0
        rho = atmosphere(segment.altitude_m or params.altitude_cruise)["rho"]
        q = 0.5 * rho * V ** 2
        CL = W / (q * S) if q > 0 else 0.5
        CD = CD0 + k * CL ** 2
        drag = q * S * CD
        fuel_rate = _fuel_rate(params, drag, V)
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
# OpenMDAO-optimized cruise speed
# ──────────────────────────────────────────────

def optimize_cruise_speed(params: AircraftParams) -> dict:
    """Use OpenMDAO SLSQP to find cruise speed minimizing fuel per distance."""
    prob = om.Problem()
    model = prob.model

    ivc = om.IndepVarComp()
    ivc.add_output('V', val=100.0, units='m/s', lower=30.0, upper=400.0)
    ivc.add_output('rho', val=atmosphere(params.altitude_cruise)['rho'],
                   units='kg/m**3')
    ivc.add_output('S', val=params.wing_area, units='m**2')
    ivc.add_output('CD0', val=params.CD0)
    ivc.add_output('k', val=params.k)
    ivc.add_output('W', val=params.MTOW * G, units='N')
    ivc.add_output('TSFC', val=params.TSFC, units='1/s')
    ivc.add_output('BSFC', val=params.BSFC, units='kg/(kW*s)')
    ivc.add_output('eta_prop', val=params.eta_prop)
    model.add_subsystem('ivc', ivc, promotes=['*'])

    model.add_subsystem('CL_calc', om.ExecComp(
        'CL = W / (0.5 * rho * V**2 * S)',
        CL={'val': 0.5}, W={'units': 'N'},
        rho={'units': 'kg/m**3'}, V={'units': 'm/s'}, S={'units': 'm**2'},
    ), promotes=['W', 'rho', 'S'])
    model.connect('V', 'CL_calc.V')

    model.add_subsystem('drag_calc', om.ExecComp(
        'D = 0.5 * rho * V**2 * S * (CD0 + k * CL**2)',
        D={'units': 'N'}, rho={'units': 'kg/m**3'},
        V={'units': 'm/s'}, S={'units': 'm**2'},
    ), promotes=['rho', 'S', 'CD0', 'k'])
    model.connect('V', 'drag_calc.V')
    model.connect('CL_calc.CL', 'drag_calc.CL')

    if params.propulsion_type == "turboprop":
        model.add_subsystem('fuel_per_dist', om.ExecComp(
            'fpd = BSFC * D / (eta_prop * 1000.0)',
            fpd={'units': 'kg/m'}, D={'units': 'N'},
            BSFC={'units': 'kg/(kW*s)'}, eta_prop={'val': 0.82},
        ), promotes=['BSFC'])
        model.connect('drag_calc.D', 'fuel_per_dist.D')
    else:
        model.add_subsystem('fuel_per_dist', om.ExecComp(
            'fpd = TSFC * D / V',
            fpd={'units': 'kg/m'}, D={'units': 'N'},
            V={'units': 'm/s'}, TSFC={'units': '1/s'},
        ), promotes=['TSFC'])
        model.connect('drag_calc.D', 'fuel_per_dist.D')
        model.connect('V', 'fuel_per_dist.V')

    model.add_objective('fuel_per_dist.fpd', ref=0.01)
    model.add_design_var('V', lower=30.0, upper=400.0, ref=200.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 200
    prob.driver.options['tol'] = 1e-9

    prob.setup()
    prob.run_model()
    prob.run_driver()

    return {
        "optimal_cruise_speed_ms": prob.get_val('V')[0],
        "optimal_cruise_CL": prob.get_val('CL_calc.CL')[0],
        "optimal_cruise_drag_N": prob.get_val('drag_calc.D')[0],
        "fuel_per_distance_kg_m": prob.get_val('fuel_per_dist.fpd')[0],
    }


def optimize_loiter_speed(params: AircraftParams) -> dict:
    """Use OpenMDAO SLSQP to find loiter speed minimizing fuel rate."""
    prob = om.Problem()
    model = prob.model

    ivc = om.IndepVarComp()
    ivc.add_output('V', val=100.0, units='m/s', lower=50.0, upper=300.0)
    ivc.add_output('rho', val=atmosphere(params.altitude_cruise)['rho'],
                   units='kg/m**3')
    ivc.add_output('S', val=params.wing_area, units='m**2')
    ivc.add_output('CD0', val=params.CD0)
    ivc.add_output('k', val=params.k)
    ivc.add_output('W', val=params.MTOW * G, units='N')
    ivc.add_output('TSFC', val=params.TSFC, units='1/s')
    ivc.add_output('BSFC', val=params.BSFC, units='kg/(kW*s)')
    ivc.add_output('eta_prop', val=params.eta_prop)
    model.add_subsystem('ivc', ivc, promotes=['*'])

    model.add_subsystem('CL_calc', om.ExecComp(
        'CL = W / (0.5 * rho * V**2 * S)',
        CL={'val': 0.5}, W={'units': 'N'},
        rho={'units': 'kg/m**3'}, V={'units': 'm/s'}, S={'units': 'm**2'},
    ), promotes=['W', 'rho', 'S'])
    model.connect('V', 'CL_calc.V')

    model.add_subsystem('drag_calc', om.ExecComp(
        'D = 0.5 * rho * V**2 * S * (CD0 + k * CL**2)',
        D={'units': 'N'}, rho={'units': 'kg/m**3'},
        V={'units': 'm/s'}, S={'units': 'm**2'},
    ), promotes=['rho', 'S', 'CD0', 'k'])
    model.connect('V', 'drag_calc.V')
    model.connect('CL_calc.CL', 'drag_calc.CL')

    if params.propulsion_type == "turboprop":
        model.add_subsystem('fuel_rate', om.ExecComp(
            'fr = BSFC * D * V / (eta_prop * 1000.0)',
            fr={'units': 'kg/s'}, D={'units': 'N'}, V={'units': 'm/s'},
            BSFC={'units': 'kg/(kW*s)'}, eta_prop={'val': 0.82},
        ), promotes=['BSFC'])
        model.connect('drag_calc.D', 'fuel_rate.D')
        model.connect('V', 'fuel_rate.V')
    else:
        model.add_subsystem('fuel_rate', om.ExecComp(
            'fr = TSFC * D',
            fr={'units': 'kg/s'}, D={'units': 'N'}, TSFC={'units': '1/s'},
        ), promotes=['TSFC'])
        model.connect('drag_calc.D', 'fuel_rate.D')

    model.add_objective('fuel_rate.fr', ref=1.0)
    model.add_design_var('V', lower=50.0, upper=300.0, ref=150.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 200
    prob.driver.options['tol'] = 1e-9

    prob.setup()
    prob.run_driver()

    fr_opt = prob.get_val('fuel_rate.fr')[0]
    return {
        "optimal_loiter_speed_ms": prob.get_val('V')[0],
        "optimal_loiter_CL": prob.get_val('CL_calc.CL')[0],
        "optimal_loiter_drag_N": prob.get_val('drag_calc.D')[0],
        "fuel_rate_kg_s": fr_opt,
        "endurance_h_per_kg": 1.0 / fr_opt / 3600.0 if fr_opt > 0 else 0,
    }
