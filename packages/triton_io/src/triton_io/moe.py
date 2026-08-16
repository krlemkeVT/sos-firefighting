"""Measure-of-effectiveness helpers for wildfire simulation outputs."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from triton_io.wildfire_outputs import get_simulation_output_path, read_json_file

INVALID_MOE = -100000.0

# These constants intentionally match the current wildfire batch script so the
# refactor preserves optimization behavior while moving the math into triton_io.
SCENARIO_MAXIMA = {
    "Salamis": {
        "BurntArea_ha": 4146.0,
        "CostArea_EURM": 13993.0,
        "Emission_ton": 714009.0,
        "FleetOps_EURk": 268.0,
    },
    "Pyrenees": {
        "BurntArea_ha": 9938.0,
        "CostArea_EURM": 17509.0,
        "Emission_ton": 2364064.0,
        "FleetOps_EURk": 167.0,
    },
    "Palisades": {
        "BurntArea_ha": 9087.0,
        "CostArea_EURM": 191106.0,
        "Emission_ton": 131224.0,
        "FleetOps_EURk": 250.0,
    },
}

MOE_WEIGHTS = (0.2, 0.2, 0.2, 0.3, 0.1)
FUEL_EUR_PER_KG = 2.0
MAINTENANCE_EUR_PER_FLEET_FLIGHT_HR = 5000.0
SUPPRESSION_WEAR_EUR_PER_DROP = 50.0
ELECTRICITY_EUR_PER_KWH = 0.25


def _write_debug_json(path: Path, data: dict[str, Any]) -> None:
    """Persist per-seed debug information when callers request it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_float(simulation_data: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric field from the simulation JSON with a forgiving fallback."""

    value = simulation_data.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def compute_moe(
    out_base: Path,
    scenario_name: str,
    fleet_acq_eur: float,
    debug_dir: Path | None = None,
    debug: bool = False,
) -> float:
    """Compute wildfire MoE from a SoSID output base path.

    The output base comes directly from ``run_sim(...)``. We intentionally read
    the adjacent ``*_simulation.json`` file instead of assuming a monolithic
    result file, because that matches the existing wildfire behavior.
    """

    if scenario_name not in SCENARIO_MAXIMA:
        raise KeyError(
            f"Unknown scenario {scenario_name!r}. "
            f"Expected one of {sorted(SCENARIO_MAXIMA)}."
        )

    simulation_path = get_simulation_output_path(out_base)
    simulation_data = read_json_file(simulation_path)
    mission_success = bool(simulation_data.get("mission_success", True))

    if not mission_success:
        if debug_dir is not None:
            _write_debug_json(
                debug_dir / f"{out_base.name}_moe_debug.json",
                {
                    "scenario": scenario_name,
                    "simulation_path": str(simulation_path),
                    "mission_success": False,
                    "note": "Returned invalid MoE because mission_success was false.",
                },
            )
        return INVALID_MOE

    burnt_area_m2 = _get_float(simulation_data, "burnt_area", 0.0)
    total_fire_cost_eur = _get_float(simulation_data, "total_fire_cost", 0.0)
    total_fire_emissions = _get_float(
        simulation_data,
        "total_fire_emissions",
        0.0,
    )
    total_fuel_kg = _get_float(simulation_data, "total_network_fuel", 0.0)
    flight_time_s = _get_float(
        simulation_data,
        "fleet_cumulative_flight_time",
        0.0,
    )
    suppressions = _get_float(
        simulation_data,
        "fleet_total_suppressions",
        0.0,
    )
    electric_energy_j = _get_float(
        simulation_data,
        "total_network_electric_energy",
        0.0,
    )

    flight_hours = flight_time_s / 3600.0
    electric_energy_kwh = electric_energy_j / 3.6e6

    fuel_cost_eur = total_fuel_kg * FUEL_EUR_PER_KG
    maintenance_cost_eur = (
        flight_hours * MAINTENANCE_EUR_PER_FLEET_FLIGHT_HR
    )
    wear_cost_eur = suppressions * SUPPRESSION_WEAR_EUR_PER_DROP
    electricity_cost_eur = electric_energy_kwh * ELECTRICITY_EUR_PER_KWH
    fleet_ops_cost_eur = (
        fuel_cost_eur
        + maintenance_cost_eur
        + wear_cost_eur
        + electricity_cost_eur
    )

    burnt_area_ha = burnt_area_m2 / 10000.0
    cost_area_eurm = total_fire_cost_eur / 1e6
    emission_ton = total_fire_emissions
    fleet_acq_eurm = float(fleet_acq_eur) / 1e6
    fleet_ops_eurk = fleet_ops_cost_eur / 1000.0

    maxima = SCENARIO_MAXIMA[scenario_name]
    w1, w2, w3, w4, w5 = MOE_WEIGHTS
    t1 = burnt_area_ha / maxima["BurntArea_ha"]
    t2 = cost_area_eurm / maxima["CostArea_EURM"]
    t3 = emission_ton / maxima["Emission_ton"]
    t4 = fleet_acq_eurm / 100.0
    t5 = fleet_ops_eurk / maxima["FleetOps_EURk"]

    c1 = w1 * t1
    c2 = w2 * t2
    c3 = w3 * t3
    c4 = w4 * t4
    c5 = w5 * t5
    penalty = c1 + c2 + c3 + c4 + c5
    moe = 1.0 - penalty

    if debug:
        logging.info("[triton_io.moe] scenario=%s simulation=%s", scenario_name, simulation_path)
        logging.info("[triton_io.moe] burnt_area_ha=%s cost_area_eurm=%s emission_ton=%s", burnt_area_ha, cost_area_eurm, emission_ton)
        logging.info("[triton_io.moe] fleet_acq_eurm=%s fleet_ops_eurk=%s", fleet_acq_eurm, fleet_ops_eurk)
        logging.info("[triton_io.moe] terms=%s", {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5})
        logging.info("[triton_io.moe] contributions=%s moe=%s", {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, moe)

    if not math.isfinite(moe):
        moe = INVALID_MOE

    if debug_dir is not None:
        _write_debug_json(
            debug_dir / f"{out_base.name}_moe_debug.json",
            {
                "scenario": scenario_name,
                "simulation_path": str(simulation_path),
                "mission_success": mission_success,
                "fleet_acq_eur": float(fleet_acq_eur),
                "raw": {
                    "burnt_area_m2": burnt_area_m2,
                    "total_fire_cost_eur": total_fire_cost_eur,
                    "total_fire_emissions": total_fire_emissions,
                    "total_network_fuel_kg": total_fuel_kg,
                    "fleet_cumulative_flight_time_s": flight_time_s,
                    "fleet_total_suppressions": suppressions,
                    "total_network_electric_energy_j": electric_energy_j,
                },
                "derived": {
                    "flight_hours": flight_hours,
                    "electric_energy_kwh": electric_energy_kwh,
                    "fuel_cost_eur": fuel_cost_eur,
                    "maintenance_cost_eur": maintenance_cost_eur,
                    "wear_cost_eur": wear_cost_eur,
                    "electricity_cost_eur": electricity_cost_eur,
                    "fleet_ops_cost_eur": fleet_ops_cost_eur,
                    "burnt_area_ha": burnt_area_ha,
                    "cost_area_eurm": cost_area_eurm,
                    "emission_ton": emission_ton,
                    "fleet_acq_eurm": fleet_acq_eurm,
                    "fleet_ops_eurk": fleet_ops_eurk,
                },
                "terms": {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5},
                "weighted_terms": {
                    "c1": c1,
                    "c2": c2,
                    "c3": c3,
                    "c4": c4,
                    "c5": c5,
                },
                "penalty": penalty,
                "moe": float(moe),
            },
        )

    return float(moe)
