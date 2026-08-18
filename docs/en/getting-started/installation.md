# Installation

MLFCS requires Python 3.12 or newer.

```bash
uv sync
```

For an existing environment:

```bash
python -m pip install .
```

The base dependencies are ASE, NumPy, SciPy, spglib, h5py, and JAX. Calculator packages and
downstream readers are intentionally optional. For example, install plotting dependencies only
when running the phonopy/SeeK-path example with `uv run --with phonopy --with seekpath`.
