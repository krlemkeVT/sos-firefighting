# TRITON

TRITON is a research monorepo for wildfire response simulation, aircraft sizing, and optimization workflows.

The repository is organized as multiple Python packages inside one repository. Each package owns a specific subsystem, while shared interfaces live in a small API package. The goal is to let contributors work on one subsystem without casually breaking the others.

## Repository Layout

```text
triton/
├── packages/
│   ├── wildfire/
│   │   └── Existing wildfire / COSID simulation package
│   │
│   ├── aircraft_sizing/
│   │   └── Aircraft sizing and design models
│   │
│   ├── optimization/
│   │   └── Optuna, DOE, orchestration, and design-space exploration
│   │
│   └── triton_api/
│       └── Shared dataclasses, protocols, and interface definitions
│
├── examples/
│   └── Small runnable examples using the packages
│
├── docs/
│   └── Architecture notes, design decisions, and project documentation
│
├── scripts/
│   └── Utility scripts
│
├── tests/
│   └── Cross-package integration tests
│
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

## Packages

### `wildfire`

The `wildfire` package contains the wildfire simulation code, including fire spread, terrain, suppression logic, firefighter models, and related simulation tools.

This package should own wildfire-specific logic only.

Expected usage:

```python
from sosid.wildfire import WildfireSimulation, SimulationConfig

config = SimulationConfig(...)
simulation = WildfireSimulation(config)
result = simulation.run()
```

The exact import path may vary depending on the upstream wildfire package structure. The goal is for reusable wildfire logic to be imported through the package API, not from `examples/`.

### `aircraft_sizing`

The `aircraft_sizing` package contains aircraft sizing, design, feasibility, and performance models.

This package should not know whether it is being called by Optuna, DOE, a GUI, a notebook, or a standalone script. It should expose clean sizing functions/classes and return structured results.

Expected usage:

```python
from aircraft_sizing import DefaultAircraftSizer
from triton_api.aircraft import AircraftDesignVariables

sizer = DefaultAircraftSizer()

design = AircraftDesignVariables(
    wing_area_m2=12.0,
    aspect_ratio=8.5,
    payload_kg=200.0,
    cruise_speed_mps=60.0,
    fuel_mass_kg=100.0,
)

result = sizer.size(design)
```

### `optimization`

The `optimization` package contains optimization and design-space exploration logic.

This includes:

- Optuna studies
- DOE generation
- objective functions
- subprocess/worker orchestration
- simulation-evaluation pipelines
- parameter sweeps
- result aggregation

The optimization package is allowed to call `wildfire`, `aircraft_sizing`, and `triton_api`.

Expected usage:

```python
from triton_optimization import run_study

run_study(n_trials=100)
```

### `triton_api`

The `triton_api` package contains shared interfaces used by the other packages.

This includes:

- dataclasses
- result objects
- protocol definitions
- shared type definitions
- interface contracts between packages

This package should stay lightweight. It should not contain heavy simulation, aircraft sizing, or optimization logic.

Example interface:

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass
class AircraftDesignVariables:
    wing_area_m2: float
    aspect_ratio: float
    payload_kg: float
    cruise_speed_mps: float
    fuel_mass_kg: float


@dataclass
class AircraftSizingResult:
    mtow_kg: float
    empty_weight_kg: float
    fuel_burn_kg: float
    range_km: float
    feasible: bool
    constraint_violations: dict[str, float]


class AircraftSizer(Protocol):
    def size(self, design: AircraftDesignVariables) -> AircraftSizingResult:
        ...
```

## Dependency Rules

The repository uses one-way package dependencies to avoid circular imports.

Allowed dependency direction:

```text
triton_api
   ↑             ↑              ↑
   |             |              |
wildfire     aircraft_sizing   optimization
                                ↑
                                |
                  calls wildfire and aircraft_sizing
```

Rules:

1. `triton_api` must not import from `wildfire`, `aircraft_sizing`, or `optimization`.
2. `aircraft_sizing` may import from `triton_api`.
3. `wildfire` may import from `triton_api`.
4. `optimization` may import from `triton_api`, `aircraft_sizing`, and `wildfire`.
5. `aircraft_sizing` must not import from `optimization`.
6. `wildfire` must not import from `optimization`.
7. Packages must not import from `examples`.
8. `examples` may import from packages.


For example, aircraft sizing may return feasibility information, constraint violations, suggested bounds, warnings, and callback updates to optimization. But aircraft sizing should not import Optuna or optimizer-specific code.

## Development Setup (NOT YET IMPLEMENTED)

Clone the repository:

```bash
git clone <repo-url>
cd triton
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the packages in editable mode:

```bash
pip install -e packages/triton_api
pip install -e packages/wildfire
pip install -e packages/aircraft_sizing
pip install -e packages/optimization
```

Install development tools:

```bash
pip install pytest ruff
```

Verify the setup:

```bash
python -c "import triton_api; import aircraft_sizing; import triton_optimization"
```

If the wildfire package exposes `cosid`, verify:

```bash
python -c "import cosid"
```

## Running Tests (NOT YET IMPLEMENTED)

Run all tests:

```bash
pytest
```

Run tests for a specific package:

```bash
pytest packages/wildfire/tests
pytest packages/aircraft_sizing/tests
pytest packages/optimization/tests
pytest packages/triton_api/tests
```

Run cross-package integration tests:

```bash
pytest tests
```

Before merging changes to `triton_api`, run the full test suite:

```bash
pytest
```

## Formatting and Linting (NOT YET IMPLEMENTED)

Format code:

```bash
ruff format .
```

Check code:

```bash
ruff check .
```

Fix automatically where possible:

```bash
ruff check . --fix
```

## Examples

Examples should be small runnable scripts that demonstrate package usage.

Good:

```python
from aircraft_sizing import DefaultAircraftSizer
from triton_api.aircraft import AircraftDesignVariables

sizer = DefaultAircraftSizer()
result = sizer.size(AircraftDesignVariables(...))

print(result)
```

Bad:

```python
from examples.wildfire.optuna.worker import ...
```

No package should depend on code inside `examples`.

If an example becomes required by another package, move that logic into the correct package.

## Architecture Philosophy

This is a monorepo, but it is not one giant package.

Each subsystem has a clear responsibility:

```text
wildfire          = wildfire simulation
aircraft_sizing   = aircraft design, sizing, and feasibility
optimization      = design-space search and orchestration
triton_api        = shared contracts and interfaces
```

Optimization coordinates the full workflow. It may call wildfire and aircraft sizing, but wildfire and aircraft sizing should remain independent of the optimizer.

This keeps the system modular, testable, and easier for multiple people to work on at the same time.

## Collaboration Model (NOT YET IMPLEMENTED)

Contributors should work on focused branches and open pull requests into `main`.

Do not commit directly to `main`.

Typical workflow:

```bash
git checkout main
git pull
git checkout -b feature/aircraft-endurance-model

# make changes

ruff format .
ruff check .
pytest packages/aircraft_sizing/tests packages/triton_api/tests

git add .
git commit -m "Add aircraft endurance model"
git push -u origin feature/aircraft-endurance-model
```

Then open a pull request.

## Minimum Success Check (NOT YET IMPLEMENTED)

A fresh environment should eventually support:

```python
from cosid.wildfire import WildfireSimulation
from aircraft_sizing import DefaultAircraftSizer
from triton_optimization import run_study
from triton_api.aircraft import AircraftDesignVariables
```

If these imports work cleanly, the package boundaries are in good shape.

## License

This repository is proprietary and not open source until released at the end of research.

No permission is granted to use, copy, modify, distribute, publish, sublicense, or commercialize this software without explicit prior written permission from all authors/copyright holders.

See [`LICENSE`](LICENSE) for details.
