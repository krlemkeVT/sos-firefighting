from __future__ import annotations

import json

from triton_api.runner import AircraftFleetEntry
from triton_io.moe import compute_moe
from triton_io.wildfire_overwrites import build_overwrites


def test_build_overwrites_replaces_agents_and_preserves_other_fields(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "name": "baseline_case",
                "agents": [{"file_name": "old.json"}],
                "nested": {
                    "keep": True,
                    "values": {
                        "unchanged": 1,
                        "overwritten": 2,
                    },
                },
            },
        ),
        encoding="utf-8",
    )

    fleet = [
        AircraftFleetEntry(
            file_name="dhc_515.json",
            agents_per_base=[1, 0],
            suppression_tactic={"main": {"select_poi": "water"}},
        ),
        AircraftFleetEntry(
            file_name="trial_aircraft.json",
            agents_per_base=[0, 3],
            suppression_tactic={"main": {"select_poi": "vip"}},
        ),
    ]
    scenario_modifiers = {
        "nested": {
            "values": {
                "overwritten": 99,
            },
        },
        "new_field": "added",
    }

    overwrites = build_overwrites(
        baseline_path,
        fleet,
        scenario_modifiers,
    )

    assert [entry["file_name"] for entry in overwrites["agents"]] == [
        "dhc_515.json",
        "trial_aircraft.json",
    ]
    assert overwrites["nested"]["keep"] is True
    assert overwrites["nested"]["values"]["unchanged"] == 1
    assert overwrites["nested"]["values"]["overwritten"] == 99
    assert overwrites["new_field"] == "added"


def test_compute_moe_raises_readable_error_for_missing_outputs(tmp_path):
    missing_out_base = tmp_path / "missing_seed" / "Palisades_out_seed0"

    try:
        compute_moe(
            missing_out_base,
            "Palisades",
            fleet_acq_eur=40_000_000.0,
        )
    except FileNotFoundError as exc:
        assert "simulation output file" in str(exc)
        assert "Nearby JSON files" in str(exc)
    else:
        raise AssertionError("Expected compute_moe to raise FileNotFoundError.")
