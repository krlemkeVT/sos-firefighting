"""Full pipeline: generate aircraft with OpenMDAO, run wildfire simulation.

This script:
1. Uses the aircraft_sizing package to generate an aircraft JSON
2. Saves it into the wildfire aircraft data directory
3. Builds a ScenarioRunRequest pointing at it
4. Runs the wildfire simulation through triton_runner

Prerequisites (install in this order):
  python -m pip install -e packages\triton_api
  python -m pip install -e packages\aircraft_sizing
  python -m pip install -e packages\wildfire
  python -m pip install -e packages\triton_io
  python -m pip install -e packages\triton_runner

Then:
  python run_full_simulation.py --execute
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aircraft_sizing import DefaultAircraftSizer
from triton_api.runner import AircraftFleetEntry, ScenarioRunRequest
from triton_runner import run_scenario


def generate_aircraft():
    """Step 1: Generate aircraft JSON using OpenMDAO sizer."""
    print("=" * 60)
    print("  Step 1: Generate aircraft with OpenMDAO sizer")
    print("=" * 60)

    sizer = DefaultAircraftSizer(preset="c172")
    result = sizer.size()

    # Save to the wildfire aircraft data directory
    aircraft_dir = Path("packages/wildfire/examples/wildfire/data/aircraft")
    aircraft_path = aircraft_dir / "openmdao_cl415.json"

    # Strip the _performance debug key for the sim
    sim_json = {k: v for k, v in result.items() if k != "_performance"}
    aircraft_path.write_text(json.dumps(sim_json, indent=2), encoding="utf-8")

    print(f"  Saved aircraft JSON to: {aircraft_path}")
    print(f"  MTOM: {result['mtom']} kg")
    print(f"  Cruise fuel: {result['propulsion_input']['cruise_fc']} kg/s")
    print(f"  Loiter fuel: {result['propulsion_input']['loiter_fc']} kg/s")

    perf = result["_performance"]
    print(f"  Optimal cruise: {perf['optimal_cruise_kt']:.0f} kt")
    print(f"  Max L/D: {perf['max_LD']}")
    print()

    return "openmdao_cl415.json"


def build_request(aircraft_file: str) -> ScenarioRunRequest:
    """Step 2: Build a scenario request using the generated aircraft."""
    print("=" * 60)
    print("  Step 2: Build scenario request")
    print("=" * 60)

    tactic = {
        "main": {
            "select_poi": "water",
            "track_poi": "direct",
            "suppress": "direct",
        },
        "alternative": {
            "change_condition": "no_change",
            "threshold": None,
            "alternative_tactic": {
                "select_poi": "water",
                "track_poi": "direct",
                "suppress": "direct",
            },
        },
    }

    request = ScenarioRunRequest(
        run_id="openmdao_palisades",
        run_dir=Path("runs/openmdao_palisades"),
        scenario_name="Palisades",
        input_file="Palisades.json",
        baseline_overwrites_file="baseline_palisades.json",
        fleet=[
            AircraftFleetEntry(
                file_name="dhc_515.json",
                agents_per_base=[1, 0],
                suppression_tactic=tactic,
            ),
            AircraftFleetEntry(
                file_name=aircraft_file,
                agents_per_base=[0, 0],
                suppression_tactic=tactic,
            ),
        ],
        scenario_modifiers={},
        seeds=[1],
        fleet_acq_eur=80_000_000.0,
        metadata={
            "note": "OpenMDAO-sized CL-415 + reference DHC-515, Palisades scenario",
        },
    )

    print(f"  Scenario: {request.scenario_name}")
    print(f"  Fleet: DHC-515 (1) + OpenMDAO CL-415 (2)")
    print(f"  Seeds: {request.seeds}")
    print()

    return request


def main():
    """Run the full pipeline."""
    aircraft_file = generate_aircraft()
    request = build_request(aircraft_file)

    if "--execute" not in sys.argv:
        print("  Pass --execute to run the wildfire simulation.")
        print("  (This takes a while — it runs the full wildfire model.)")
        return 0

    print("=" * 60)
    print("  Step 3: Run wildfire simulation")
    print("=" * 60)

    result = run_scenario(request)

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Feasible:  {result.feasible}")
    print(f"  Mean MoE:  {result.mean_moe}")
    print(f"  Errors:    {result.errors}")

    if result.per_seed:
        print()
        for seed_result in result.per_seed:
            print(f"  Seed {seed_result.get('seed')}: "
                  f"feasible={seed_result.get('feasible')}, "
                  f"MoE={seed_result.get('moe')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
