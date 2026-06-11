# Contributing to TRITON

This document describes how to contribute to the TRITON monorepo without breaking other packages or creating circular dependencies.

TRITON is organized as multiple packages in one repository. Contributors should respect package ownership, keep pull requests focused, and avoid editing shared APIs casually.

## Repository Structure

```text
triton/
├── packages/
│   ├── wildfire/
│   ├── aircraft_sizing/
│   ├── optimization/
│   └── triton_api/
├── examples/
├── docs/
├── scripts/
└── tests/
```

## Package Responsibilities

### `packages/wildfire`

Owns wildfire simulation logic.

This includes:

- fire spread models
- terrain logic
- suppression/firefighter models
- simulation state
- simulation result generation

This package should not import from `optimization`.

### `packages/aircraft_sizing`

Owns aircraft design, sizing, and feasibility logic.

This includes:

- sizing methods
- performance models
- feasibility checks
- constraint calculations
- aircraft result objects

This package should not import from `optimization`.

### `packages/optimization`

Owns optimization and orchestration logic.

This includes:

- Optuna studies
- DOE generation
- parameter sweeps
- objective functions
- worker/subprocess logic
- integration of wildfire and aircraft sizing results

This package may import from `wildfire`, `aircraft_sizing`, and `triton_api`.

### `packages/triton_api`

Owns shared contracts and interfaces.

This includes:

- dataclasses
- protocols
- shared result types
- interface definitions between packages

This package must remain lightweight and must not import from the other packages.

## Dependency Rules

The dependency direction is:

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

Two-way data flow is allowed. Two-way imports are not.

If aircraft sizing needs to communicate information back to optimization, return richer result objects or use callbacks defined in `triton_api`. Do not make aircraft sizing import optimization.

## Branching

Do not commit directly to `main`.

Use short, descriptive branches:

```text
feature/wildfire-spread-model
feature/aircraft-sizing-api
feature/optimization-parallel-workers
fix/api-result-contract
docs/repo-architecture
```

Avoid vague branch names:

```text
updates
changes
final
aarav-work
big-refactor
```

Keep branches focused and short-lived.

A good branch usually changes one package. A branch that changes three packages should have a clear reason and should be reviewed carefully.

## Pull Requests

All changes should go through pull requests.

A pull request should include:

- a clear title
- a short explanation of what changed
- the package or packages affected
- test results
- notes about any API changes

Example PR description:

```text
## Summary

Adds endurance feasibility checks to aircraft sizing.

## Changed Packages

- aircraft_sizing
- triton_api

## Tests

- pytest packages/aircraft_sizing/tests
- pytest packages/triton_api/tests

## Notes

Adds `endurance_min` to `AircraftDesignVariables`.
```

## Pull Request Rules

Before requesting review:

```bash
ruff format .
ruff check .
pytest
```

If the change only affects one package, run at least that package's tests and any relevant `triton_api` tests.

Examples:

For aircraft sizing changes:

```bash
pytest packages/aircraft_sizing/tests packages/triton_api/tests
```

For optimization changes:

```bash
pytest packages/optimization/tests packages/triton_api/tests
```

For wildfire changes:

```bash
pytest packages/wildfire/tests packages/triton_api/tests
```

For `triton_api` changes:

```bash
pytest
```

## Shared API Changes

Changes to `triton_api` affect every package.

Do not casually rename fields, delete dataclasses, or change result types. That will break other people's work.

Bad:

```python
# Old
result.feasible

# New, breaking change
result.is_valid
```

Better:

```python
@dataclass
class AircraftSizingResult:
    feasible: bool
    is_valid: bool | None = None
```

Then update callers gradually and remove the old field later after coordination.

If a shared interface needs to change, open an issue or design note first unless the change is tiny.

## Examples Policy

The `examples/` folder is for small runnable demos only.

Examples may import from packages.

Packages must not import from examples.

Good:

```python
from aircraft_sizing import DefaultAircraftSizer
from triton_api.aircraft import AircraftDesignVariables
```

Bad:

```python
from examples.wildfire.optuna.worker import ...
```

If code in `examples/` becomes reusable or necessary for another package, move it into the correct package.

## Testing

Each package should have its own tests:

```text
packages/wildfire/tests
packages/aircraft_sizing/tests
packages/optimization/tests
packages/triton_api/tests
```

Cross-package tests should go in:

```text
tests/
```

Run all tests:

```bash
pytest
```

Run package-specific tests:

```bash
pytest packages/aircraft_sizing/tests
```

## Formatting and Linting

Use Ruff for formatting and linting.

Format:

```bash
ruff format .
```

Check:

```bash
ruff check .
```

Fix automatically where possible:

```bash
ruff check . --fix
```

Do not submit PRs with unrelated formatting changes across the whole repository unless the PR is specifically for formatting.

## Commit Messages

Use clear commit messages.

Good:

```text
Add aircraft endurance constraint
Refactor optimization objective interface
Fix wildfire terrain loading path
Add API contract for aircraft sizing result
```

Bad:

```text
fix
updates
stuff
final
please work
```

## Code Ownership

Package ownership should be defined in `.github/CODEOWNERS`.

Example:

```text
/packages/wildfire/           @wildfire-team
/packages/aircraft_sizing/    @aircraft-team
/packages/optimization/       @optimization-team
/packages/triton_api/         @core-api-team
/examples/                    @core-api-team
/docs/                        @core-api-team
```

If GitHub teams are not set up, use individual usernames.

Changes to `triton_api` should receive extra review because they affect all packages.

## Local Development Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install packages in editable mode:

```bash
pip install -e packages/triton_api
pip install -e packages/wildfire
pip install -e packages/aircraft_sizing
pip install -e packages/optimization
```

Install dev tools:

```bash
pip install pytest ruff
```

Verify imports:

```bash
python -c "import triton_api; import aircraft_sizing; import triton_optimization"
```

If the wildfire package exposes `cosid`, verify:

```bash
python -c "import cosid"
```

## Cross-Package Changes

Cross-package changes should be split when possible.

Preferred sequence:

1. Update `triton_api`.
2. Update the package implementing the new API.
3. Update `optimization` or examples to consume the new API.
4. Remove deprecated code after everything works.

Avoid giant PRs that refactor wildfire, aircraft sizing, optimization, and API contracts all at once.

## Licensing and Contribution Rights

This repository is proprietary and not open source.

By contributing to this repository, you certify that:

1. you have the legal right to submit the contribution;
2. your contribution does not knowingly include code or content you are not allowed to submit;
3. your contribution is submitted under the repository's proprietary license terms;
4. you understand that no public license is granted by this repository;
5. external use, copying, modification, distribution, publication, sublicensing, or commercialization requires explicit prior written permission from all authors/copyright holders.

If the project requires a formal Contributor License Agreement or copyright assignment, that must be handled separately in writing. This file is not a substitute for legal advice.

## Security and Sensitive Information

Do not commit:

- API keys
- private credentials
- access tokens
- proprietary third-party files without permission
- personal information
- large generated artifacts unless required
- local environment files

Use `.gitignore` for generated outputs, local virtual environments, caches, and secrets.

## Questions

If a change affects multiple packages or changes a shared interface, ask before implementing it. Guessing on shared architecture is how monorepos turn into a mess.
