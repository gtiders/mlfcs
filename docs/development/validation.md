# Numerical validation

Numerical validation is split between deterministic local tests and material cases. The local
tests check mathematical identities, structure mappings, sparse support, writer contracts, and
small public workflows. They are not run by CI.

The analytic Morse FC4 test is an internal independent-energy check. It verifies finite-difference
order and FC4 values without relying on another force-constant implementation.

Material comparisons formerly kept under `tests/reference/` now belong to the corresponding
`examples/<Material>/` README. They retain third-party outputs and provenance as human-
reviewed evidence rather than automatic truth arrays. A comparison must state primitive and
supercell conventions, atom order, units, cutoff, periodic-image convention, and constraints
before reporting IFC, phonon, or transport differences.

Confidence is built from force errors, invariants, aligned IFC or dynamical-matrix comparisons,
phonon stability and convergence, NAC settings, and transport convergence. Similar observables do
not prove a writer correct when upstream structures or q-space conventions differ.

## Local checks

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

## Documentation CI

CI only builds both documentation sites with strict link and navigation checking. It does not run
Python tests, third-party calculators, scientific benchmarks, or package builds.
