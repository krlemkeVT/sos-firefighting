"""Scenario and batch runners for wildfire simulations.

This module intentionally owns orchestration only:
- build generated overwrite files,
- execute wildfire runs,
- parse outputs,
- compute MoE,
- and return standardized result objects.

It does not contain optimization logic, aircraft sizing logic, or wildfire
simulation internals.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from triton_api.runner import (
    BatchRunRequest,
    BatchRunResult,
    ScenarioRunRequest,
    ScenarioRunResult,
)
from triton_io import (
    build_overwrites,
    collect_output_files,
    compute_moe,
    write_overwrites,
)

RUN_SIM_OUT_BASE_PREFIX = "__TRITON_RUN_SIM_OUT_BASE__:"


def _serialize_for_json(value: Any) -> Any:
    """Convert Paths inside nested structures into JSON-safe strings."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]
    return value


def _resolve_wildfire_inputs_path(file_name_or_path: str) -> Path:
    """Resolve wildfire scenario and overwrite inputs.

    If the caller already provided a real path, we use it as-is. Otherwise we
    fall back to the wildfire example scenario directory so existing filename
    usage like ``Palisades.json`` and ``baseline_palisades.json`` keeps working.
    """

    candidate = Path(file_name_or_path)
    if candidate.exists():
        return candidate.resolve()

    from examples.wildfire.paths import SCENARIOS_DIR

    return (SCENARIOS_DIR / "inputs" / file_name_or_path).resolve()


def _run_sim_subprocess(
    *,
    input_file: Path,
    overwrites_file: Path,
    output_dir: Path,
    seed: int,
    force_headless: bool = True,
    python_exe: str | None = None,
) -> Path:
    """Run the wildfire simulation in a fresh Python process.

    We preserve subprocess isolation because the current Optuna workflow uses it
    to avoid cross-trial state leakage in the wildfire example code.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    python_command = python_exe or sys.executable

    # TODO: move this import target into the installed sosid package API instead
    # of importing from examples.wildfire.main once wildfire exposes a stable
    # public runner entrypoint.
    launcher = f"""
import sys
from examples.wildfire.main import run_sim

out_base = run_sim(
    input_file=sys.argv[1],
    overwrites_file=sys.argv[2],
    output_dir=sys.argv[3],
    seed=int(sys.argv[4]),
    force_headless=(sys.argv[5] == "1"),
)
print("{RUN_SIM_OUT_BASE_PREFIX}" + str(out_base))
""".strip()

    command = [
        python_command,
        "-c",
        launcher,
        str(input_file),
        str(overwrites_file),
        str(output_dir),
        str(seed),
        "1" if force_headless else "0",
    ]

    logging.info(
        "[triton_runner] launching wildfire subprocess for seed %s into %s",
        seed,
        output_dir,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )

    assert process.stdout is not None
    out_base: Path | None = None
    for line in process.stdout:
        stripped = line.rstrip("\n")
        if stripped.startswith(RUN_SIM_OUT_BASE_PREFIX):
            out_base = Path(
                stripped.removeprefix(RUN_SIM_OUT_BASE_PREFIX),
            ).resolve()
            logging.info(
                "[triton_runner] captured output base for seed %s: %s",
                seed,
                out_base,
            )
        else:
            logging.info("[triton_runner][seed=%s] %s", seed, stripped)

    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Wildfire subprocess failed for seed {seed} with return code "
            f"{return_code}."
        )
    if out_base is None:
        raise RuntimeError(
            f"Wildfire subprocess completed for seed {seed}, but did not "
            "report an output base path."
        )
    return out_base


def _write_summary(path: Path, data: dict[str, Any]) -> Path:
    """Write runner summaries after converting Paths to strings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_serialize_for_json(data), indent=2),
        encoding="utf-8",
    )
    return path


def run_scenario(request: ScenarioRunRequest) -> ScenarioRunResult:
    """Execute one scenario across the requested seeds.

    Callers receive a structured result instead of needing to know about the
    wildfire file layout, seed loops, or MoE calculation details.
    """

    scenario_dir = request.run_dir / request.scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    per_seed: list[dict[str, Any]] = []
    output_files: dict[str, Path] = {}
    mean_moe: float | None = None

    if request.fleet_acq_eur is None:
        errors.append(
            "ScenarioRunRequest.fleet_acq_eur is required to compute wildfire MoE.",
        )
    if not request.seeds:
        errors.append("ScenarioRunRequest.seeds must contain at least one seed.")

    try:
        input_path = _resolve_wildfire_inputs_path(request.input_file)
        baseline_path = _resolve_wildfire_inputs_path(
            request.baseline_overwrites_file,
        )
        generated_overwrites = build_overwrites(
            baseline_path,
            request.fleet,
            request.scenario_modifiers,
        )
        overwrites_path = write_overwrites(
            scenario_dir / "overwrites.json",
            generated_overwrites,
        )
        output_files["overwrites_json"] = overwrites_path
    except Exception as exc:
        errors.append(
            f"Failed to prepare scenario inputs for {request.scenario_name}: {exc}",
        )
        input_path = Path(request.input_file)
        overwrites_path = scenario_dir / "overwrites.json"

    successful_moes: list[float] = []
    if not errors:
        for seed in request.seeds:
            seed_output_dir = scenario_dir / f"seed_{seed}" / "outputs"
            seed_record: dict[str, Any] = {
                "seed": seed,
                "output_dir": str(seed_output_dir),
            }
            try:
                out_base = _run_sim_subprocess(
                    input_file=input_path,
                    overwrites_file=overwrites_path,
                    output_dir=seed_output_dir,
                    seed=seed,
                    force_headless=True,
                )
                output_files[f"seed_{seed}_out_base"] = out_base
                for name, path in collect_output_files(out_base).items():
                    output_files[f"seed_{seed}_{name}"] = path

                moe = compute_moe(
                    out_base,
                    request.scenario_name,
                    fleet_acq_eur=float(request.fleet_acq_eur),
                )
                successful_moes.append(moe)
                seed_record.update(
                    {
                        "feasible": True,
                        "moe": moe,
                        "out_base": str(out_base),
                    },
                )
            except Exception as exc:
                error_message = (
                    f"Seed {seed} failed for scenario "
                    f"{request.scenario_name}: {exc}"
                )
                errors.append(error_message)
                seed_record.update(
                    {
                        "feasible": False,
                        "error": error_message,
                    },
                )

            per_seed.append(seed_record)

        if not errors and successful_moes:
            mean_moe = sum(successful_moes) / len(successful_moes)

    summary = {
        "run_id": request.run_id,
        "scenario_name": request.scenario_name,
        "input_file": request.input_file,
        "baseline_overwrites_file": request.baseline_overwrites_file,
        "fleet_acq_eur": request.fleet_acq_eur,
        "seeds": request.seeds,
        "metadata": request.metadata,
        "per_seed": per_seed,
        "mean_moe": mean_moe,
        "errors": errors or None,
    }
    summary_path = _write_summary(scenario_dir / "scenario_summary.json", summary)
    output_files["scenario_summary_json"] = summary_path

    feasible = not errors and mean_moe is not None
    metrics: dict[str, float] = {}
    if mean_moe is not None:
        metrics["mean_moe"] = float(mean_moe)

    return ScenarioRunResult(
        run_id=request.run_id,
        scenario_name=request.scenario_name,
        feasible=feasible,
        mean_moe=mean_moe if feasible else None,
        per_seed=per_seed,
        metrics=metrics,
        output_files=output_files,
        errors=errors or None,
    )


def run_batch(request: BatchRunRequest) -> BatchRunResult:
    """Execute a sequential batch of scenarios.

    The runner deliberately stays single-threaded by default so optimizers, DOE
    drivers, and HPC schedulers keep control over outer parallelism.
    """

    request.batch_dir.mkdir(parents=True, exist_ok=True)

    results: list[ScenarioRunResult] = []
    errors: list[str] = []
    output_files: dict[str, Path] = {}
    feasible_means: list[float] = []

    for scenario_request in request.scenarios:
        result = run_scenario(scenario_request)
        results.append(result)
        output_files[f"{result.scenario_name}_summary_json"] = (
            result.output_files["scenario_summary_json"]
        )
        if result.errors:
            errors.extend(result.errors)
        if result.feasible and result.mean_moe is not None:
            feasible_means.append(result.mean_moe)

    metrics: dict[str, float] = {}
    if feasible_means:
        metrics["mean_moe"] = sum(feasible_means) / len(feasible_means)

    summary = {
        "batch_id": request.batch_id,
        "metadata": request.metadata,
        "scenario_results": [
            {
                "scenario_name": result.scenario_name,
                "feasible": result.feasible,
                "mean_moe": result.mean_moe,
                "errors": result.errors,
            }
            for result in results
        ],
        "metrics": metrics,
        "errors": errors or None,
    }
    batch_summary_path = _write_summary(
        request.batch_dir / "batch_summary.json",
        summary,
    )
    output_files["batch_summary_json"] = batch_summary_path

    return BatchRunResult(
        batch_id=request.batch_id,
        results=results,
        metrics=metrics,
        output_files=output_files,
        errors=errors or None,
    )
