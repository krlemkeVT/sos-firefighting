"""
Script used to run the Wildfire example and launch the SoSID Viewer.

Arguments can be passed to WildfireParameters in order to replace default values.

This file is now Optuna-friendly:
- You can import and call run_sim(...) without auto-running the sim.
- Output path is unique per run (seed in name) to avoid collisions on HPC.
- overwrites_file can be either:
    - a filename in SCENARIOS_DIR/inputs (e.g., "baseline_palisades.json"), OR
    - an absolute/relative path to a JSON overwrites file (e.g., Optuna trial folder).
"""

from pathlib import Path

from pyinstrument import Profiler

from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.simulation import WildfireParameters, WildfireSimulation
from sosid.gui.main import display
from sosid.output import OutputFormat
from sosid.util.general_funcs import combine_parameters


def run_sim(
    input_file: str = "Salamis.json",
    overwrites_file: str | None = "baseline_salamis.json",
    output_dir: str | Path | None = None,
    seed: int = 0,
    force_headless: bool = True,
    run_profiler: bool = False,
) -> Path:
    """
    Run a single instance of the wildfire sim.

    Args:
        input_file: Scenario file name in SCENARIOS_DIR/inputs (e.g. "Palisades.json")
        overwrites_file: Baseline/tactics/aircraft modifications. Either:
            - a filename in SCENARIOS_DIR/inputs (e.g. "baseline_palisades.json"), or
            - a path to a JSON file (e.g. "/.../trial_0001/overwrites.json"), or
            - None for no overwrites.
        output_dir: Where to write outputs. If None, uses SCENARIOS_DIR/outputs.
        seed: Simulation seed (Optuna usually sets this to trial.number).
        force_headless: If True, forces run_headless so it never opens GUI.
        run_profiler: If True, runs with pyinstrument profiler.

    Returns:
        out_base: Base path used for output writing (SoSID will add .json, etc.)
    """

    input_filepath = SCENARIOS_DIR / "inputs" / input_file

    # Resolve overwrites path:
    # - If overwrites_file is a plain name, treat it as living in inputs/
    # - If it looks like a path that exists, use it directly
    overwrite_path: Path | None = None
    if overwrites_file:
        candidate = Path(overwrites_file)
        if candidate.exists():
            overwrite_path = candidate
        else:
            overwrite_path = (SCENARIOS_DIR / "inputs") / overwrites_file

        params_dict = combine_parameters(input_filepath, overwrite_path)
        parameters = WildfireParameters.model_validate(params_dict)
    else:
        parameters = WildfireParameters.model_validate_json(input_filepath.read_text())

    # Force headless for optimization runs (avoid GUI)
    if force_headless:
        try:
            parameters.run_headless = True
        except Exception:
            # If pydantic model is frozen/immutable
            parameters = parameters.model_copy(update={"run_headless": True})

    sim = WildfireSimulation(
        parameters=parameters,
        seed=seed,
        context=Profiler(interval=0.001) if run_profiler else None,
    )

    if run_profiler:
        sim.start()
        sim.join()
        sim.context.open_in_browser()
    elif parameters.run_headless:
        print("Running Headless")
        sim.start()
        sim.is_stopped.wait()
    else:
        display(sim)

    # Decide output directory
    if output_dir is None:
        output_dir = SCENARIOS_DIR / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unique output base name (prevents collisions on HPC)
    out_base = output_dir / f"{input_filepath.stem}_out_seed{seed}"

    # Write outputs if terminated
    if sim.is_stopped.is_set():
        sim.write_output_data(
            sim.get_output_data(),
            out_base,
            OutputFormat.JSON,
        )

    return out_base


if __name__ == "__main__":
    # Default manual run (keeps your original behavior)
    out = run_sim(
        input_file="Palisades.json",
        overwrites_file="baseline_palisades.json",
        seed=0,
        force_headless=False,  # set True if you never want GUI here
        run_profiler=False,
    )
    print(f"Wrote output base to: {out}")
