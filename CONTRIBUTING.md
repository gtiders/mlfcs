# Contributing to MLFCS

[中文](CONTRIBUTING_ZH.md)

Bug reports, reference datasets, documentation fixes, and focused pull requests are welcome.

## Before opening an issue

Search existing issues and reduce the problem to a reproducible example. Include the MLFCS,
Python, ASE, NumPy, SciPy, and spglib versions; operating system; primitive structure and cell;
orders, cutoffs, supercell and displacement settings; and the full traceback or numerical
comparison. Do not attach proprietary potentials or calculations unless redistribution is allowed.

## Development setup

```bash
git clone https://github.com/gtiders/mlfcs.git
cd mlfcs
uv sync --group dev --group reference
```

Run the numerical suite and independent reference comparisons:

```bash
uv run pytest -m "not reference"
uv run pytest -m reference
uv run ruff check src tests
uv build
```

The reference group supplies SymPy for independent exact-algebra comparisons. The tutorial
notebooks under `docs/notebooks/` call phonopy or phono3py and use the optional tutorial group:

```bash
uv sync --group tutorial
```

Documentation contributors can install the docs group, check the Chinese pages and tutorial
notebooks, and build the site with `uv sync --group docs`, `uv run python docs/scripts/check_docs.py`,
and `uv run mkdocs build --strict -f mkdocs.yml`.

## Tests and tutorials

Keep numerical tests focused on reproducible behavior. Exact algebra and scientific results should
be compared with independent reference implementations where available. Do not add permanent tests
for `mlfcs.tools`; validate tool changes with a temporary check that is not committed. Teaching fits
live in the tutorial notebooks: their committed execution outputs are the result record, and
re-execution is serial, one notebook at a time.

Tutorial calculations may require separately prepared structures, external executables, or large
reference datasets. Document those inputs and their provenance; do not add files unless they may be
redistributed.

## Pull requests

Keep changes scoped and explain their scientific or API motivation. Update the Chinese
documentation when public behavior changes. Preserve unrelated worktree changes, avoid committing
build products, and add a changelog entry for user-visible changes. Contributions are accepted
under the GNU General Public License v3.0 or later.
