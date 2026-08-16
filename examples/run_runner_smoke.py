"""Small smoke example for the refactored wildfire runner API.

This example is intentionally lightweight:
- it proves the packaged imports and request construction work,
- it documents the expected editable installs,
- and it only runs the expensive wildfire simulation when ``--execute`` is
  provided explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

from triton_api.runner import AircraftFleetEntry, ScenarioRunRequest
from triton_runner import run_scenario


def build_request() -> ScenarioRunRequest:
    """Build a small one-seed scenario request using existing wildfire assets."""

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

    return ScenarioRunRequest(
        run_id="manual_palisades_smoke",
        run_dir=Path("runs/manual_palisades_smoke"),
        scenario_name="Palisades",
        input_file="Palisades.json",
        baseline_overwrites_file="baseline_palisades.json",
        fleet=[
            AircraftFleetEntry(
                file_name="dhc_515.json",
                agents_per_base=[1, 0],
                suppression_tactic=tactic,
            ),
        ],
        scenario_modifiers={},
        seeds=[0],
        fleet_acq_eur=40_000_000.0,
        metadata={
            "note": (
                "Install packages/wildfire, packages/triton_api, "
                "packages/triton_io, and packages/triton_runner in editable "
                "mode before executing this example."
            ),
        },
    )


def main() -> int:
    """Build the request and optionally execute it."""

    request = build_request()
    print(f"Constructed request for scenario {request.scenario_name}: {request}")

    if "--execute" not in sys.argv:
        print("Skipping the expensive wildfire run. Pass --execute to run it.")
        return 0

    result = run_scenario(request)
    print(f"Scenario feasible: {result.feasible}")
    print(f"Scenario mean MoE: {result.mean_moe}")
    print(f"Scenario errors: {result.errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
