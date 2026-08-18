# Local tests

The `tests/` directory contains only deterministic unit and public-API regression tests. All test
files are flat and named `test_<area>_<behavior>.py`.

Run them locally with:

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

Tests use minimal structures, fixed random seeds, and tolerances justified by units or numerical
analysis. They do not read material cases, depend on external software, or use developer paths.

Material calculations and third-party comparisons live under `examples/`. Their README files
describe commands and results; they are not automated test oracles.
