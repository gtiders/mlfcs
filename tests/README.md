# Testing

Tests cover supported computations and their numerical results. They do not
inspect package layering or source syntax, assert that removed APIs are absent,
or deliberately exercise invalid inputs and failure handling. Production
validation and error handling continue to operate independently of this scope.

## Running tests

Install the development dependencies and run the ordinary suite:

```bash
uv sync --group dev
uv run --group dev pytest -m "not reference"
```

For the scientific oracle comparisons, install the optional reference group:

```bash
uv sync --group dev --group reference
uv run --group dev --group reference pytest -m reference
```

Run both sets with `pytest` in an environment containing the required packages.
Missing optional oracle packages skip their individual reference tests; ordinary
matrix, reconstruction and zero-constraint tests remain available.

## Organization

Files follow the current features: algebra, geometry, cluster space, candidates,
mapping, force design, fitting, finite differences, force constants and exports,
invariance projections, reciprocal calculations, SCPH, and logging. Tools use
temporary validation at implementation time and have no standing tests.

Integer results use elementwise equality. Floating comparisons specify tolerances;
invariant subspaces use projection matrices and integer lattices use HNF or Smith
invariant factors. Random inputs use fixed seeds. Parameterized IDs identify the
material, tensor order or individual matrix rather than hiding a corpus in one loop.

The `data/` fixtures and `oracles/` implementations retain independent scientific
references. Small synthetic force formulas isolate signs, factorials, physical
units and solver behavior. Tests of private numerical kernels use independent
formulas or reference implementations for the expected results.
