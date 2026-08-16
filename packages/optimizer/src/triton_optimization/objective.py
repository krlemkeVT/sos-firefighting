"""Optuna objective helpers for aircraft sizing plus wildfire evaluation.

This module keeps the orchestration thin:
- sample design variables,
- run aircraft sizing,
- stage the selected aircraft,
- build wildfire runner requests,
- and delegate execution to :mod:`triton_runner`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from triton_api.runner import (
    AircraftFleetEntry,
    BatchRunRequest,
    ScenarioRunRequest,
)
from triton_runner import run_batch

SCENARIOS = [
    ("Salamis", "Salamis.json"),
    ("Pyrenees", "Pyrenees.json"),
    ("Palisades", "Palisades.json"),
]

BASELINE_OVERWRITES_BY_SCENARIO = {
    "Salamis": "baseline_salamis.json",
    "Pyrenees": "baseline_pyrenees.json",
    "Palisades": "baseline_palisades.json",
}

FIXED_SEEDS = [0, 1, 2]

MAX_MTOM_KG = 25_000.0
BUDGET_TOTAL_EUR = 100_000_000.0
TANKER_COST_EUR = 40_000_000.0
BUDGET_LEFT_EUR = BUDGET_TOTAL_EUR - TANKER_COST_EUR
MASS_TOLERANCE_KG = 200.0

SELECT_OPTIONS = ["water", "vip", "vegetation", "topography", "indirect"]
TRACK_OPTIONS = ["direct", "indirect", "follow_firefront"]
SUPPRESS_OPTIONS = ["direct", "indirect"]
CHANGE_OPTIONS = [
    "no_change",
    "runtime",
    "daytime",
    "residential",
    "burnt_area",
    "distance",
]


def _raise_trial_pruned() -> None:
    """Raise Optuna's prune signal lazily so tests can import this module easily."""

    import optuna

    raise optuna.TrialPruned()


def _require_env_path(name: str, default: str | None = None) -> Path:
    """Read a required environment variable as a resolved path."""

    value = os.environ.get(name, default)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required.")
    return Path(value).expanduser().resolve()


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    """Write a debug-friendly JSON artifact into the trial directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def unit_cost_eur(
    mtom_kg: float,
    n_units: int,
    ratio_composite: float,
    ratio_aluminum: float,
    ratio_steeltitan: float,
) -> float:
    """Preserve the existing Optuna fleet cost model exactly."""

    b = 0.7
    learn_rate = 0.9
    complexity_factor = 1.3

    material_factor = (
        ratio_composite * 1.35
        + ratio_aluminum * 1.0
        + ratio_steeltitan * 1.2
    )

    mtom_ref = 17000.0
    cost_ref = 60e6
    learning_exponent = math.log(learn_rate, 2)

    coefficient = cost_ref / (
        (mtom_ref**b)
        * complexity_factor
        * material_factor
        * (1.0**learning_exponent)
    )

    base_cost = (
        coefficient
        * (mtom_kg**b)
        * (float(n_units) ** learning_exponent)
        * complexity_factor
        * material_factor
    )
    return base_cost * 0.85


def sample_aircraft_design(trial: Any) -> dict[str, float]:
    """Sample the aircraft design variables and material ratios for one trial."""

    mtow = trial.suggest_float("max_takeoff_mass_kg", 5000, 20000)
    design = {
        "max_takeoff_mass_kg": mtow,
        "payload_kg": trial.suggest_float("payload_kg", 1000, 0.8 * mtow),
        "wingspan_m": trial.suggest_float("wingspan_m", 10, 20),
        "stall_speed_m_p_s": trial.suggest_float("stall_speed_m_p_s", 45, 65),
        "rotor_diameter_m": trial.suggest_float("rotor_diameter_m", 1.5, 5.0),
        "tip_mach": trial.suggest_float("tip_mach", 0.7, 0.85),
        "ratio_composite": trial.suggest_float("ratio_composite", 0.3, 0.6),
        "ratio_aluminum": trial.suggest_float("ratio_aluminum", 0.2, 0.7),
    }
    design["ratio_steeltitan"] = (
        1.0 - design["ratio_composite"] - design["ratio_aluminum"]
    )
    if design["ratio_steeltitan"] <= 0:
        _raise_trial_pruned()
    return design


def _try_float(value: Any) -> Any:
    """Cast numeric strings to floats while leaving non-numeric strings alone."""

    if isinstance(value, str):
        try:
            if value.strip() == "":
                return value
            return float(value)
        except Exception:
            return value
    return value


def _recursive_numeric_cast(obj: Any) -> Any:
    """Normalize aircraft JSON output so downstream code sees numeric values."""

    if isinstance(obj, dict):
        return {key: _recursive_numeric_cast(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_recursive_numeric_cast(value) for value in obj]
    return _try_float(obj)


def normalize_aircraft_json_inplace(path: Path) -> None:
    """Normalize sized aircraft JSON exactly like the current Optuna workflow."""

    aircraft_data = json.loads(path.read_text(encoding="utf-8"))
    aircraft_data = _recursive_numeric_cast(aircraft_data)

    propulsion_input = aircraft_data.get("propulsion_input")
    if isinstance(propulsion_input, dict) and "cruise_fc" in propulsion_input:
        cruise_fc = propulsion_input["cruise_fc"]
        if isinstance(cruise_fc, dict):
            normalized_cruise_fc = {}
            for key, value in cruise_fc.items():
                try:
                    normalized_cruise_fc[int(float(key))] = float(value)
                except Exception:
                    normalized_cruise_fc[key] = value
            propulsion_input["cruise_fc"] = normalized_cruise_fc

    path.write_text(json.dumps(aircraft_data, indent=2), encoding="utf-8")


def run_aircraft_sizing_for_trial(
    trial_dir: Path,
    design_variables: dict[str, float],
) -> tuple[Path, dict[str, Any]]:
    """Run the external aircraft sizing flow and return the produced JSON."""

    aircraft_cfg_template = _require_env_path("AIRCRAFT_CFG_TEMPLATE")
    aircraft_run_dir = _require_env_path("AIRCRAFT_RUN_DIR")
    aircraft_run_script = aircraft_run_dir / os.environ.get(
        "AIRCRAFT_RUN_SCRIPT",
        "update_and_run.py",
    )

    template_data = json.loads(aircraft_cfg_template.read_text(encoding="utf-8"))
    template_data["aircraft"]["max_takeoff_mass_kg"] = design_variables[
        "max_takeoff_mass_kg"
    ]
    template_data["aircraft"]["payload_kg"] = design_variables["payload_kg"]
    template_data["aircraft"]["wingspan_m"] = design_variables["wingspan_m"]
    template_data["aircraft"]["stall_speed_m_p_s"] = design_variables[
        "stall_speed_m_p_s"
    ]
    template_data["aircraft"]["rotor_diameter_m"] = design_variables[
        "rotor_diameter_m"
    ]
    template_data["aircraft"]["tip_mach"] = design_variables["tip_mach"]

    cfg_path = _write_json(trial_dir / "aircraft_cfg.json", template_data)

    subprocess.run(
        [
            sys.executable,
            str(aircraft_run_script),
            str(cfg_path),
            str(trial_dir),
            str(trial_dir),
        ],
        cwd=str(aircraft_run_dir),
        check=True,
    )

    produced_files = sorted(
        (
            path
            for path in trial_dir.glob("*.json")
            if path.name != "aircraft_cfg.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not produced_files:
        raise RuntimeError(
            f"Aircraft sizing did not produce a JSON result in {trial_dir}.",
        )

    produced_path = produced_files[-1]
    normalize_aircraft_json_inplace(produced_path)
    return produced_path, json.loads(produced_path.read_text(encoding="utf-8"))


def validate_and_fix_sized_aircraft(
    aircraft_path: Path,
    aircraft_data: dict[str, Any],
) -> dict[str, Any]:
    """Apply the existing MTOM tolerance logic before wildfire evaluation."""

    mtom = float(aircraft_data.get("mtom", 0.0))
    empty_mass = float(aircraft_data.get("empty_mass", 0.0))
    payload_mass = float(aircraft_data.get("payload", 0.0))

    propellant_mass = 0.0
    propulsion_input = aircraft_data.get("propulsion_input", {})
    if isinstance(propulsion_input, dict):
        propellant_mass = float(propulsion_input.get("total_propellant", 0.0))

    total_mass = empty_mass + payload_mass + propellant_mass
    mass_error = total_mass - mtom

    if abs(mass_error) <= MASS_TOLERANCE_KG:
        aircraft_data["mtom"] = total_mass
        aircraft_path.write_text(
            json.dumps(aircraft_data, indent=2),
            encoding="utf-8",
        )
        mtom = total_mass
    elif total_mass > mtom:
        _raise_trial_pruned()

    if mtom <= 0 or mtom > MAX_MTOM_KG:
        _raise_trial_pruned()

    return aircraft_data


def compute_affordable_fleet_size(
    mtom_kg: float,
    ratio_composite: float,
    ratio_aluminum: float,
    ratio_steeltitan: float,
) -> tuple[int, float]:
    """Preserve the iterative fleet sizing logic from the original objective."""

    best_n = 0
    best_unit_cost = 0.0
    for n_units in range(1, 501):
        candidate_unit_cost = unit_cost_eur(
            mtom_kg,
            n_units,
            ratio_composite,
            ratio_aluminum,
            ratio_steeltitan,
        )
        if n_units * candidate_unit_cost <= BUDGET_LEFT_EUR:
            best_n = n_units
            best_unit_cost = candidate_unit_cost
        else:
            break

    if best_n < 1:
        _raise_trial_pruned()

    return best_n, best_unit_cost


def stage_aircraft_for_wildfire(
    aircraft_path: Path,
    trial_number: int,
) -> tuple[str, Path]:
    """Copy the sized aircraft JSON into the wildfire aircraft directory."""

    wildfire_aircraft_dir = _require_env_path("WILDFIRE_AIRCRAFT_DIR")
    wildfire_aircraft_dir.mkdir(parents=True, exist_ok=True)

    aircraft_name = f"aircraft_trial_{trial_number:06d}.json"
    staged_path = wildfire_aircraft_dir / aircraft_name
    shutil.copy2(aircraft_path, staged_path)
    return aircraft_name, staged_path


def build_tactic(trial: Any, prefix: str) -> dict[str, Any]:
    """Preserve the current categorical search space for one tactic block."""

    return {
        "select_poi": trial.suggest_categorical(
            f"{prefix}_select",
            SELECT_OPTIONS,
        ),
        "track_poi": trial.suggest_categorical(
            f"{prefix}_track",
            TRACK_OPTIONS,
        ),
        "suppress": trial.suggest_categorical(
            f"{prefix}_suppress",
            SUPPRESS_OPTIONS,
        ),
    }


def build_change_block(trial: Any) -> dict[str, Any]:
    """Preserve the alternative tactic search space from the original study."""

    condition = trial.suggest_categorical("change_condition", CHANGE_OPTIONS)

    if condition == "no_change":
        threshold: Any = None
    elif condition == "runtime":
        threshold = trial.suggest_int("runtime_threshold_hr", 1, 10)
    elif condition == "daytime":
        start = trial.suggest_int("day_start", 0, 20)
        end = trial.suggest_int("day_end", start + 1, 24)
        threshold = [start, end]
    elif condition == "residential":
        threshold = trial.suggest_int(
            "residential_threshold_m",
            50,
            500,
            step=100,
        )
    elif condition == "burnt_area":
        threshold = trial.suggest_int(
            "burnt_area_threshold",
            5000,
            200000,
            step=1000,
        )
    else:
        threshold = trial.suggest_int(
            "distance_threshold_m",
            50,
            500,
            step=100,
        )

    return {
        "change_condition": condition,
        "threshold": threshold,
        "alternative_tactic": build_tactic(trial, "alt"),
    }


def build_suppression_tactic(trial: Any) -> dict[str, Any]:
    """Build the nested tactic structure expected by wildfire overwrites."""

    return {
        "main": build_tactic(trial, "main"),
        "alternative": build_change_block(trial),
    }


def build_batch_request_for_trial(
    trial_number: int,
    trial_dir: Path,
    aircraft_name: str,
    aircraft_count: int,
    suppression_tactic: dict[str, Any],
    fleet_acq_eur: float,
) -> BatchRunRequest:
    """Build the standardized batch request consumed by triton_runner."""

    fleet = [
        AircraftFleetEntry(
            file_name="dhc_515.json",
            agents_per_base=[1, 0],
            suppression_tactic=suppression_tactic,
        ),
        AircraftFleetEntry(
            file_name=aircraft_name,
            agents_per_base=[0, aircraft_count],
            suppression_tactic=suppression_tactic,
        ),
    ]

    scenarios = [
        ScenarioRunRequest(
            run_id=f"trial_{trial_number:06d}_{scenario_name.lower()}",
            run_dir=trial_dir,
            scenario_name=scenario_name,
            input_file=input_file,
            baseline_overwrites_file=BASELINE_OVERWRITES_BY_SCENARIO[scenario_name],
            fleet=fleet,
            scenario_modifiers={},
            seeds=list(FIXED_SEEDS),
            fleet_acq_eur=fleet_acq_eur,
            metadata={"trial_number": trial_number},
        )
        for scenario_name, input_file in SCENARIOS
    ]

    return BatchRunRequest(
        batch_id=f"trial_{trial_number:06d}",
        batch_dir=trial_dir,
        scenarios=scenarios,
        metadata={"trial_number": trial_number},
    )


def objective(trial: Any) -> float:
    """Run one Optuna objective evaluation through aircraft sizing and runner."""

    run_root = _require_env_path("OPTUNA_RUN_DIR")
    trial_dir = run_root / "optuna_trials" / f"trial_{trial.number:06d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    design_variables = sample_aircraft_design(trial)
    aircraft_path, aircraft_data = run_aircraft_sizing_for_trial(
        trial_dir,
        design_variables,
    )
    aircraft_data = validate_and_fix_sized_aircraft(aircraft_path, aircraft_data)

    aircraft_name, staged_path = stage_aircraft_for_wildfire(
        aircraft_path,
        trial.number,
    )

    mtom = float(aircraft_data["mtom"])
    best_n, best_unit_cost = compute_affordable_fleet_size(
        mtom,
        design_variables["ratio_composite"],
        design_variables["ratio_aluminum"],
        design_variables["ratio_steeltitan"],
    )
    fleet_acq_eur = (best_n * best_unit_cost) + TANKER_COST_EUR

    _write_json(
        trial_dir / "cost_info.json",
        {
            "mtom_kg": mtom,
            "unit_cost_eur": best_unit_cost,
            "n_optuna": best_n,
            "ratio_composite": design_variables["ratio_composite"],
            "ratio_aluminum": design_variables["ratio_aluminum"],
            "ratio_steeltitan": design_variables["ratio_steeltitan"],
            "fleet_acq_eur": fleet_acq_eur,
            "staged_aircraft_path": str(staged_path),
        },
    )

    suppression_tactic = build_suppression_tactic(trial)
    batch_request = build_batch_request_for_trial(
        trial.number,
        trial_dir,
        aircraft_name,
        best_n,
        suppression_tactic,
        fleet_acq_eur,
    )
    batch_result = run_batch(batch_request)

    if batch_result.errors or any(not result.feasible for result in batch_result.results):
        logging.warning(
            "Pruning trial %s because batch execution reported errors: %s",
            trial.number,
            batch_result.errors,
        )
        _raise_trial_pruned()

    mean_moe = batch_result.metrics.get("mean_moe")
    if mean_moe is None:
        _raise_trial_pruned()

    return float(mean_moe)
