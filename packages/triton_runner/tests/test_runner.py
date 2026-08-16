from __future__ import annotations

import json
import math
from pathlib import Path

from triton_api.runner import (
    AircraftFleetEntry,
    BatchRunRequest,
    BatchRunResult,
    ScenarioRunRequest,
    ScenarioRunResult,
)
from triton_runner.runner import run_batch, run_scenario


def _build_request(tmp_path: Path) -> ScenarioRunRequest:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "Palisades.json"
    baseline_path = tmp_path / "baseline_palisades.json"
    input_path.write_text("{}", encoding="utf-8")
    baseline_path.write_text(
        json.dumps({"agents": [], "name": "baseline"}),
        encoding="utf-8",
    )

    return ScenarioRunRequest(
        run_id="scenario_test",
        run_dir=tmp_path / "runs",
        scenario_name="Palisades",
        input_file=str(input_path),
        baseline_overwrites_file=str(baseline_path),
        fleet=[
            AircraftFleetEntry(
                file_name="dhc_515.json",
                agents_per_base=[1, 0],
                suppression_tactic={"main": {"select_poi": "water"}},
            ),
        ],
        scenario_modifiers={},
        seeds=[0],
        fleet_acq_eur=40_000_000.0,
    )


def test_run_batch_averages_scenario_means(monkeypatch, tmp_path):
    requests = [
        _build_request(tmp_path / "case_a"),
        _build_request(tmp_path / "case_b"),
    ]
    requests[0].scenario_name = "Salamis"
    requests[1].scenario_name = "Pyrenees"

    def fake_run_scenario(request: ScenarioRunRequest) -> ScenarioRunResult:
        mean_moe = 0.2 if request.scenario_name == "Salamis" else 0.4
        summary_path = request.run_dir / request.scenario_name / "scenario_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("{}", encoding="utf-8")
        return ScenarioRunResult(
            run_id=request.run_id,
            scenario_name=request.scenario_name,
            feasible=True,
            mean_moe=mean_moe,
            per_seed=[],
            metrics={"mean_moe": mean_moe},
            output_files={"scenario_summary_json": summary_path},
        )

    monkeypatch.setattr("triton_runner.runner.run_scenario", fake_run_scenario)

    result = run_batch(
        BatchRunRequest(
            batch_id="batch_test",
            batch_dir=tmp_path / "batch",
            scenarios=requests,
        ),
    )

    assert result.errors is None
    assert math.isclose(result.metrics["mean_moe"], 0.3)


def test_run_scenario_returns_controlled_error_when_outputs_are_missing(
    monkeypatch,
    tmp_path,
):
    request = _build_request(tmp_path)

    def fake_run_sim_subprocess(**_: object) -> Path:
        out_base = tmp_path / "runs" / "Palisades" / "seed_0" / "outputs" / "Palisades_out_seed0"
        out_base.parent.mkdir(parents=True, exist_ok=True)
        return out_base

    monkeypatch.setattr(
        "triton_runner.runner._run_sim_subprocess",
        fake_run_sim_subprocess,
    )

    result = run_scenario(request)

    assert result.feasible is False
    assert result.mean_moe is None
    assert result.errors is not None
    assert "Missing simulation output file" in result.errors[0]
    assert result.output_files["scenario_summary_json"].exists()
