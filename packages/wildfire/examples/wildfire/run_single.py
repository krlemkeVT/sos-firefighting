#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

from examples.wildfire.main import run_sim


def _setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_three_seeds.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logging.info(f"Logging to {log_path}")


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def _resolve_overwrites_path(input_arg: str, overwrites_arg: str) -> Path:
    """
    Allows:
      --overwrites /abs/path/overwrites.json
    OR
      --overwrites overwrites.json  (assumed next to --input file path)
    """
    p = Path(overwrites_arg)
    if p.is_absolute():
        return p

    input_path = Path(input_arg)
    input_dir = input_path.parent if str(input_path.parent) != "." else Path.cwd()
    return (input_dir / overwrites_arg).resolve()


def compute_moe_with_fleet_acq(
    out_base: Path,
    scenario: str,
    *,
    fleet_acq_eur: float,
    debug_dir: Path | None = None,
    debug: bool = False,
) -> float:
    """
    Local MoE calculator (does NOT touch moe_gen.py).

    - Loads <out_base.name>_simulation.json next to out_base
    - Uses provided fleet_acq_eur for term4 (no inference from agents)
    - If debug=True:
        * prints EVERY raw/derived value used BEFORE breakdown
        * prints normalized terms + weighted contributions
        * writes per-seed debug JSON into debug_dir (if provided)
    """
    # --- MoE weights ---
    w1, w2, w3, w4, w5 = 0.2, 0.2, 0.2, 0.3, 0.1

    # --- Scenario normalization maxima (match your moe_gen numbers) ---
    scenario_max = {
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
    if scenario not in scenario_max:
        raise KeyError(f"Unknown scenario {scenario!r}. Choose from {list(scenario_max)}")

    # --- Ops cost assumptions (match moe_gen placeholders) ---
    fuel_eur_per_kg = 2.0
    maintenance_eur_per_fleet_flight_hr = 5000.0
    suppression_wear_eur_per_drop = 50.0
    electricity_eur_per_kwh = 0.25

    # ---- locate & load sim json ----
    sim_path = out_base.parent / f"{out_base.name}_simulation.json"
    if not sim_path.exists():
        nearby = sorted(out_base.parent.glob("*_simulation.json"))
        raise FileNotFoundError(
            f"Missing simulation json: {sim_path}\nNearby candidates: {[p.name for p in nearby]}"
        )

    sim = _read_json(sim_path)

    # ---- mission success gate ----
    mission_success = bool(sim.get("mission_success", True))
    if not mission_success:
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                debug_dir / f"{out_base.name}_moe_debug.json",
                {
                    "sim_path": str(sim_path),
                    "mission_success": False,
                    "note": "Returned INVALID (-100000) due to mission_success==False",
                },
            )
        return -100000.0

    # ---- pull raw fields (log defaults if missing) ----
    def getf(key: str, default: float = 0.0) -> float:
        v = sim.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    burnt_area_m2 = getf("burnt_area", 0.0)
    total_fire_cost_eur = getf("total_fire_cost", 0.0)
    total_fire_emissions = getf("total_fire_emissions", 0.0)

    total_fuel_kg = getf("total_network_fuel", 0.0)
    flight_time_s = getf("fleet_cumulative_flight_time", 0.0)
    suppressions = getf("fleet_total_suppressions", 0.0)
    elec_j = getf("total_network_electric_energy", 0.0)

    # ---- derived ops cost pieces ----
    flight_hr = flight_time_s / 3600.0
    elec_kwh = elec_j / 3.6e6

    fuel_cost_eur = total_fuel_kg * fuel_eur_per_kg
    maint_cost_eur = flight_hr * maintenance_eur_per_fleet_flight_hr
    wear_cost_eur = suppressions * suppression_wear_eur_per_drop
    elec_cost_eur = elec_kwh * electricity_eur_per_kwh

    ops_cost_eur = fuel_cost_eur + maint_cost_eur + wear_cost_eur + elec_cost_eur
    fleet_ops_eurk = ops_cost_eur / 1000.0

    # ---- convert to moe term units ----
    burnt_area_ha = burnt_area_m2 / 10000.0
    cost_area_eurm = total_fire_cost_eur / 1e6
    emission_ton = total_fire_emissions
    fleet_acq_eurm = float(fleet_acq_eur) / 1e6

    # ✅ Print ALL values used (first), then breakdown
    if debug:
        logging.info("[moe_local] ===== VALUES USED FOR MOE =====")
        logging.info(f"[moe_local] scenario={scenario}")
        logging.info(f"[moe_local] sim_path={sim_path}")
        logging.info(f"[moe_local] mission_success={mission_success}")

        logging.info("[moe_local] ---- raw sim fields ----")
        logging.info(f"[moe_local] burnt_area_m2={burnt_area_m2}")
        logging.info(f"[moe_local] total_fire_cost_eur={total_fire_cost_eur}")
        logging.info(f"[moe_local] total_fire_emissions={total_fire_emissions}")
        logging.info(f"[moe_local] total_network_fuel_kg={total_fuel_kg}")
        logging.info(f"[moe_local] fleet_cumulative_flight_time_s={flight_time_s}")
        logging.info(f"[moe_local] fleet_total_suppressions={suppressions}")
        logging.info(f"[moe_local] total_network_electric_energy_j={elec_j}")

        logging.info("[moe_local] ---- fixed cost constants ----")
        logging.info(f"[moe_local] fuel_eur_per_kg={fuel_eur_per_kg}")
        logging.info(f"[moe_local] maintenance_eur_per_fleet_flight_hr={maintenance_eur_per_fleet_flight_hr}")
        logging.info(f"[moe_local] suppression_wear_eur_per_drop={suppression_wear_eur_per_drop}")
        logging.info(f"[moe_local] electricity_eur_per_kwh={electricity_eur_per_kwh}")
        logging.info(f"[moe_local] fleet_acq_eur(input)={fleet_acq_eur}")

        logging.info("[moe_local] ---- derived intermediates ----")
        logging.info(f"[moe_local] flight_hr={flight_hr}")
        logging.info(f"[moe_local] elec_kwh={elec_kwh}")
        logging.info(f"[moe_local] fuel_cost_eur={fuel_cost_eur}")
        logging.info(f"[moe_local] maint_cost_eur={maint_cost_eur}")
        logging.info(f"[moe_local] wear_cost_eur={wear_cost_eur}")
        logging.info(f"[moe_local] elec_cost_eur={elec_cost_eur}")
        logging.info(f"[moe_local] ops_cost_eur_total={ops_cost_eur}")
        logging.info(f"[moe_local] fleet_ops_eurk={fleet_ops_eurk}")

        logging.info("[moe_local] ---- converted units for terms ----")
        logging.info(f"[moe_local] burnt_area_ha={burnt_area_ha}")
        logging.info(f"[moe_local] cost_area_eurm={cost_area_eurm}")
        logging.info(f"[moe_local] emission_ton={emission_ton}")
        logging.info(f"[moe_local] fleet_acq_eurm={fleet_acq_eurm}")
        logging.info("[moe_local] =================================")

    # ---- Normalize & compute MoE ----
    mx = scenario_max[scenario]
    t1 = burnt_area_ha / mx["BurntArea_ha"]
    t2 = cost_area_eurm / mx["CostArea_EURM"]
    t3 = emission_ton / mx["Emission_ton"]
    t4 = fleet_acq_eurm / 100.0
    t5 = fleet_ops_eurk / mx["FleetOps_EURk"]

    c1, c2, c3, c4, c5 = (w1 * t1), (w2 * t2), (w3 * t3), (w4 * t4), (w5 * t5)
    penalty = c1 + c2 + c3 + c4 + c5
    moe = 1.0 - penalty

    if debug:
        logging.info(
            f"[moe_local] norm_terms: t1={t1:.6f}, t2={t2:.6f}, t3={t3:.6f}, t4={t4:.6f}, t5={t5:.6f}"
        )
        logging.info(
            f"[moe_local] weighted: c1={c1:.6f}, c2={c2:.6f}, c3={c3:.6f}, c4={c4:.6f}, c5={c5:.6f} "
            f"| penalty={penalty:.6f} moe={moe:.6f}"
        )

    if not math.isfinite(moe):
        if debug:
            logging.warning("[moe_local] Non-finite moe computed; returning INVALID (-100000)")
        moe = -100000.0

    # ---- write per-seed debug artifact ----
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            debug_dir / f"{out_base.name}_moe_debug.json",
            {
                "scenario": scenario,
                "sim_path": str(sim_path),
                "mission_success": mission_success,
                "fleet_acq_eur": float(fleet_acq_eur),
                "raw": {
                    "burnt_area_m2": burnt_area_m2,
                    "total_fire_cost_eur": total_fire_cost_eur,
                    "total_fire_emissions": total_fire_emissions,
                    "total_network_fuel_kg": total_fuel_kg,
                    "fleet_cumulative_flight_time_s": flight_time_s,
                    "fleet_total_suppressions": suppressions,
                    "total_network_electric_energy_j": elec_j,
                },
                "constants": {
                    "fuel_eur_per_kg": fuel_eur_per_kg,
                    "maintenance_eur_per_fleet_flight_hr": maintenance_eur_per_fleet_flight_hr,
                    "suppression_wear_eur_per_drop": suppression_wear_eur_per_drop,
                    "electricity_eur_per_kwh": electricity_eur_per_kwh,
                },
                "derived": {
                    "flight_hr": flight_hr,
                    "elec_kwh": elec_kwh,
                    "fuel_cost_eur": fuel_cost_eur,
                    "maint_cost_eur": maint_cost_eur,
                    "wear_cost_eur": wear_cost_eur,
                    "elec_cost_eur": elec_cost_eur,
                    "ops_cost_eur_total": ops_cost_eur,
                    "fleet_ops_eurk": fleet_ops_eurk,
                    "burnt_area_ha": burnt_area_ha,
                    "cost_area_eurm": cost_area_eurm,
                    "emission_ton": emission_ton,
                    "fleet_acq_eurm": fleet_acq_eurm,
                },
                "norm_terms": {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5},
                "weighted": {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5},
                "penalty": penalty,
                "moe": float(moe),
            },
        )

    return float(moe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Path to scenario input json (you manually edit this before running).",
    )
    ap.add_argument(
        "--overwrites",
        required=True,
        help="Absolute path to overwrites.json OR a filename assumed next to --input",
    )
    ap.add_argument("--scenario", default="Salamis", help="Scenario name for MoE normalization")
    ap.add_argument("--fleet-acq-eur", type=float, required=True, help="Fleet acquisition cost in EUR")
    ap.add_argument("--out", required=True, help="Output directory for this run")
    ap.add_argument("--seeds", default="0,1,2", help="Comma list of seeds (default: 0,1,2)")
    ap.add_argument("--headless", action="store_true", default=False, help="Force headless sim")
    ap.add_argument("--debug-moe", action="store_true", help="Print all used values + write moe_debug jsons")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    _setup_logging(out_dir)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input json not found: {input_path}")

    overwrites_path = _resolve_overwrites_path(str(input_path), args.overwrites)
    if not overwrites_path.exists():
        raise FileNotFoundError(f"overwrites.json not found: {overwrites_path}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    if len(seeds) < 1:
        raise ValueError("No seeds provided")

    per_seed: List[Dict[str, Any]] = []
    moes: List[float] = []

    logging.info(f"input_file={input_path}")
    logging.info(f"overwrites_file={overwrites_path}")
    logging.info(f"scenario={args.scenario}")
    logging.info(f"fleet_acq_eur={args.fleet_acq_eur:.3f}")
    logging.info(f"seeds={seeds}")
    logging.info(f"debug_moe={bool(args.debug_moe)}")

    debug_dir = (out_dir / "moe_debug") if args.debug_moe else None

    for seed in seeds:
        seed_outdir = out_dir / f"seed_{seed}" / "outputs"
        seed_outdir.mkdir(parents=True, exist_ok=True)

        logging.info(f"[seed {seed}] running sim -> {seed_outdir}")
        out_base = run_sim(
            input_file=str(input_path),
            overwrites_file=str(overwrites_path),
            output_dir=seed_outdir,
            seed=seed,
            force_headless=bool(args.headless),
        )

        moe = float(
            compute_moe_with_fleet_acq(
                Path(out_base),
                args.scenario,
                fleet_acq_eur=float(args.fleet_acq_eur),
                debug_dir=debug_dir,
                debug=bool(args.debug_moe),
            )
        )
        moes.append(moe)
        mean_so_far = sum(moes) / len(moes)

        logging.info(f"[seed {seed}] moe={moe:.6f} mean_so_far={mean_so_far:.6f}")
        per_seed.append({"seed": seed, "moe": moe, "output_base": str(out_base)})

    mean_moe = sum(moes) / len(moes)
    summary = {
        "input_file": str(input_path),
        "overwrites_file": str(overwrites_path),
        "scenario": args.scenario,
        "fleet_acq_eur": float(args.fleet_acq_eur),
        "seeds": seeds,
        "per_seed": per_seed,
        "mean_moe": mean_moe,
    }

    summary_path = out_dir / "trial_summary.json"
    _write_json(summary_path, summary)
    logging.info(f"DONE mean_moe={mean_moe:.6f}")
    logging.info(f"Summary => {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())