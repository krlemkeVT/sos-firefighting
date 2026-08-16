from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any


# ============================================================
# LOGGING SETUP
# ============================================================

def _setup_moe_logger(trial_dir: Path) -> logging.Logger:
    log_dir = trial_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "moe.log"

    logger = logging.getLogger(f"MOE_{trial_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.propagate = False

    logger.info("===== MOE COMPUTATION START =====")
    logger.info(f"Logging to {log_path}")

    return logger


# ============================================================
# HELPERS (You must already have these somewhere)
# ============================================================

def _find_trial_dir_from_outbase(out_base: Path) -> Path:
    for parent in out_base.parents:
        if parent.name.startswith("trial_"):
            return parent
    raise RuntimeError("Could not locate trial directory from output path.")


def _read_cost_info(trial_dir: Path) -> Dict[str, Any]:
    cost_file = trial_dir / "cost_info.json"
    if not cost_file.exists():
        raise RuntimeError("Missing cost_info.json")
    return json.loads(cost_file.read_text())


# ============================================================
# MAIN MOE
# ============================================================

def compute_moe(out_base: str | Path, scenario_name: str) -> float:

    out_base = Path(out_base).resolve()
    trial_dir = _find_trial_dir_from_outbase(out_base)
    logger = _setup_moe_logger(trial_dir)

    try:
        logger.info(f"Scenario: {scenario_name}")
        logger.info(f"Output base: {out_base}")

        # --------------------------------------------------------
        # Load main simulation output
        # --------------------------------------------------------

        output_dir = out_base.parent
        prefix = out_base.name  # "Salamis_out_seed0"

        sim_matches = list(output_dir.glob(f"{prefix}_simulation.json"))
        if not sim_matches:
            raise RuntimeError(f"No simulation file found for prefix {prefix}")

        sim_file = sim_matches[0]
        logger.info(f"Simulation file: {sim_file}")

        sim = json.loads(sim_file.read_text())

        mission_success = bool(sim.get("mission_success", True))
        logger.info(f"Mission success: {mission_success}")

        burnt_area = float(sim["burnt_area"])
        fire_cost = float(sim["total_fire_cost"])
        emissions = float(sim["total_fire_emissions"])

        logger.info(f"Burnt area: {burnt_area}")
        logger.info(f"Fire cost: {fire_cost}")
        logger.info(f"Emissions: {emissions}")

        # --------------------------------------------------------
        # Load UAV file
        # --------------------------------------------------------

        
        uav_matches = list(output_dir.glob(f"{prefix}_agents_SuppressionUAV.json"))
        if not uav_matches:
            raise RuntimeError(f"No UAV file found for prefix {prefix}")

        uav_file = uav_matches[0]
        logger.info(f"UAV file: {uav_file}")

        uav = json.loads(uav_file.read_text())

        total_propellant = 0.0
        total_energy = 0.0

        for k, v in uav.items():
            if k.endswith("total_propellant_mass_consumed"):
                total_propellant += float(v)
            if k.endswith("total_energy_consumed"):
                total_energy += float(v)

        logger.info(f"Total propellant: {total_propellant}")
        logger.info(f"Total energy: {total_energy}")

        # Cost coefficients
        FUEL_COST_PER_KG = 1.2
        ENERGY_COST_PER_J = 1.5e-10

        fleet_ops_cost = (
            total_propellant * FUEL_COST_PER_KG +
            total_energy * ENERGY_COST_PER_J
        )

        logger.info(f"Fleet ops cost: {fleet_ops_cost}")

        # --------------------------------------------------------
        # Acquisition cost
        # --------------------------------------------------------

        cost_info = _read_cost_info(trial_dir)

        fleet_acq_cost =( (
            float(cost_info["unit_cost_eur"]) *
            float(cost_info["n_optuna"])
        ) +40_000_000.0) / 1e6

        logger.info(f"Fleet acquisition cost (M€): {fleet_acq_cost}")

        # --------------------------------------------------------
        # Scenario maximum values
        # --------------------------------------------------------

        MAX_VALUES = {
            "Salamis": {
                "burnt": 4146,
                "cost": 13993,
                "emission": 714009,
                "ops": 268,
            },
            "Pyrenees": {
                "burnt": 9938,
                "cost": 17509,
                "emission": 2364064,
                "ops": 167,
            },
            "Palisades": {
                "burnt": 9087,
                "cost": 191106,
                "emission": 131224,
                "ops": 250,
            },
        }

        max_vals = MAX_VALUES[scenario_name]

        # --------------------------------------------------------
        # Weights
        # --------------------------------------------------------

        w1, w2, w3, w4, w5 = 0.2, 0.2, 0.2, 0.3, 0.1

        # --------------------------------------------------------
        # Individual terms (LOGGED SEPARATELY)
        # --------------------------------------------------------
        burnt_area_ha = burnt_area / 10000.0
        fire_cost_m = fire_cost / 1e6
        fleet_ops_cost_k = fleet_ops_cost / 1000.0

        term1 = w1 * (burnt_area_ha / max_vals["burnt"])
        term2 = w2 * (fire_cost_m / max_vals["cost"])
        term3 = w3 * (emissions / max_vals["emission"])
        term4 = w4 * (fleet_acq_cost / 100.0)
        term5 = w5 * (fleet_ops_cost_k / max_vals["ops"])

        logger.info(f"Term1 (burnt): {term1}")
        logger.info(f"Term2 (cost): {term2}")
        logger.info(f"Term3 (emissions): {term3}")
        logger.info(f"Term4 (acq): {term4}")
        logger.info(f"Term5 (ops): {term5}")

        moe = 1 - (term1 + term2 + term3 + term4 + term5)

        logger.info(f"FINAL MOE: {moe}")
        logger.info("===== MOE COMPUTATION END =====")

        for handler in logger.handlers:
            handler.flush()

        return float(moe)

    except Exception as e:
        logger.error("MOE computation failed!")
        logger.error(str(e))
        logger.error(traceback.format_exc())

        for handler in logger.handlers:
            handler.flush()

        raise