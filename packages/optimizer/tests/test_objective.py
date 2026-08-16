from __future__ import annotations

import importlib
from pathlib import Path

from triton_api.runner import BatchRunResult, ScenarioRunResult
from triton_optimization.objective import build_suppression_tactic, compute_affordable_fleet_size

objective_module = importlib.import_module("triton_optimization.objective")


class FakeTrial:
    def __init__(self) -> None:
        self.number = 7

    def suggest_float(self, name: str, low: float, high: float) -> float:
        values = {
            "max_takeoff_mass_kg": 12000.0,
            "payload_kg": 3000.0,
            "wingspan_m": 15.0,
            "stall_speed_m_p_s": 55.0,
            "rotor_diameter_m": 3.0,
            "tip_mach": 0.8,
            "ratio_composite": 0.4,
            "ratio_aluminum": 0.3,
        }
        return values[name]

    def suggest_categorical(self, name: str, options: list[str]) -> str:
        mapping = {
            "main_select": "water",
            "main_track": "direct",
            "main_suppress": "direct",
            "change_condition": "no_change",
            "alt_select": "water",
            "alt_track": "direct",
            "alt_suppress": "direct",
        }
        return mapping.get(name, options[0])

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        step: int | None = None,
    ) -> int:
        return low


def test_build_suppression_tactic_preserves_expected_shape():
    tactic = build_suppression_tactic(FakeTrial())

    assert set(tactic) == {"main", "alternative"}
    assert set(tactic["main"]) == {"select_poi", "track_poi", "suppress"}
    assert set(tactic["alternative"]) == {
        "change_condition",
        "threshold",
        "alternative_tactic",
    }


def test_compute_affordable_fleet_size_respects_budget_limit():
    n_units, unit_cost = compute_affordable_fleet_size(12000.0, 0.4, 0.3, 0.3)

    assert n_units >= 1
    assert (n_units * unit_cost) <= 60_000_000.0


def test_objective_uses_runner_batch_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("OPTUNA_RUN_DIR", str(tmp_path / "optuna_runs"))

    captured = {}

    def fake_run_aircraft_sizing_for_trial(trial_dir: Path, design_variables: dict[str, float]):
        aircraft_path = trial_dir / "sized_aircraft.json"
        aircraft_path.parent.mkdir(parents=True, exist_ok=True)
        aircraft_path.write_text("{}", encoding="utf-8")
        return aircraft_path, {"mtom": 12000.0, "empty_mass": 5000.0, "payload": 3000.0, "propulsion_input": {"total_propellant": 1000.0}}

    def fake_validate_and_fix_sized_aircraft(aircraft_path: Path, aircraft_data: dict[str, float]):
        return {"mtom": 12000.0}

    def fake_stage_aircraft_for_wildfire(aircraft_path: Path, trial_number: int):
        return "aircraft_trial_000007.json", aircraft_path

    def fake_compute_affordable_fleet_size(*args: object):
        return 4, 5_000_000.0

    def fake_run_batch(request):
        captured["request"] = request
        return BatchRunResult(
            batch_id=request.batch_id,
            results=[
                ScenarioRunResult(
                    run_id="a",
                    scenario_name="Salamis",
                    feasible=True,
                    mean_moe=0.75,
                    metrics={"mean_moe": 0.75},
                ),
            ],
            metrics={"mean_moe": 0.75},
        )

    monkeypatch.setattr(
        objective_module,
        "run_aircraft_sizing_for_trial",
        fake_run_aircraft_sizing_for_trial,
    )
    monkeypatch.setattr(
        objective_module,
        "validate_and_fix_sized_aircraft",
        fake_validate_and_fix_sized_aircraft,
    )
    monkeypatch.setattr(
        objective_module,
        "stage_aircraft_for_wildfire",
        fake_stage_aircraft_for_wildfire,
    )
    monkeypatch.setattr(
        objective_module,
        "compute_affordable_fleet_size",
        fake_compute_affordable_fleet_size,
    )
    monkeypatch.setattr(objective_module, "run_batch", fake_run_batch)

    score = objective_module.objective(FakeTrial())

    assert score == 0.75
    assert captured["request"].batch_id == "trial_000007"
    assert len(captured["request"].scenarios) == 3
    assert captured["request"].scenarios[0].fleet[0].file_name == "dhc_515.json"
    assert (
        Path(tmp_path / "optuna_runs" / "optuna_trials" / "trial_000007" / "cost_info.json").exists()
    )
