# Contributing to MLFCS

[中文](CONTRIBUTING_ZH.md)

Bug reports, reference datasets, documentation fixes, and focused pull requests are welcome.

## Before opening an issue

Search existing issues and reduce the problem to a reproducible example. Include:

- MLFCS, Python, ASE, NumPy, and JAX versions;
- operating system and CPU/GPU backend;
- primitive structure, supercell, order, cutoff, displacement, and ASR setting;
- the complete traceback or numerical comparison;
- whether forces came from `run()` or an external `sow()` / `reap()` workflow.

Do not attach proprietary potentials or calculations unless you are allowed to redistribute them.

## Development setup

```bash
git clone https://github.com/gtiders/mlfcs.git
cd mlfcs
uv sync --locked --dev
```

The public API and fast suite must pass before a pull request:

```bash
uv run ruff check src tests reference_tools
uv run ruff format --check src tests reference_tools
uv run pytest -m "not reference"
uv build
```

The repository no longer bundles the external scientific-reference fixtures.

## Test expectations

- Unit tests cover deterministic mathematical and I/O behavior.
- Integration tests use only public APIs.
- Scientific claims require an independent reference, provenance, units, atom-order mapping,
  tolerances, and a separate CI step.
- Large reference files must have clear redistribution terms and checksums.
- Tests must not depend on the legacy MLFCS implementation.

## Pull requests

Keep changes scoped and explain the scientific or API motivation. Update both English and Chinese
user documentation when public behavior changes. Preserve unrelated worktree changes, avoid
committing generated build products, and add a changelog entry for user-visible changes.

Contributions are accepted under the repository's GNU General Public License v3.0 or later.
