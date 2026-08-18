# Tests and examples

MLFCS has two distinct local validation surfaces:

| Location | Role | CI |
|---|---|---|
| `tests/` | Deterministic unit and public API regression tests | Not run in CI |
| `examples/` | Material workflows and third-party reference evidence | Not run in CI |

## Tests

All test files are flat and use `test_<area>_<behavior>.py`. Tests use minimal structures, fixed
random seeds, and unit-justified tolerances. They do not read cases, use external software, or
depend on developer paths.

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

The analytic Morse FC4 test is an internal mathematical oracle, not an external material benchmark.
Material comparisons and transport numbers are never ordinary pytest truth arrays.

## Examples and cases

Top-level `examples/*.py` files demonstrate one public API task. Material data lives in:

```text
examples/<Material>/<case>/
  README.md
  structures/
  fitting/
  finite_difference/
  results/
  observables/
```

Each README records structures, atom order, units, cutoffs, calculator/software versions, commands,
output roles, downstream q meshes and known differences. Third-party results remain reference
evidence in their native format. Cases do not require a manifest, checksum gate, or check script.

Fitting data uses strict ASE-readable `extxyz`; finite-difference data retains its ordered sow
workspace and `mlfcs-plan.json`. The reference atom order is authoritative in both workflows.

The documentation CI only builds the English and Chinese documentation sites. Scientific cases are
run manually when regenerating or reviewing a result.
