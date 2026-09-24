# MLFCS

[![Documentation](https://github.com/gtiders/mlfcs/actions/workflows/ci.yml/badge.svg)](https://github.com/gtiders/mlfcs/actions/workflows/ci.yml)
[![Documentation site](https://img.shields.io/badge/docs-GitHub%20Pages-0f766e)](https://gtiders.github.io/mlfcs/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776ab)](https://www.python.org/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

English | [简体中文](README.zh-CN.md)

<!-- BEGIN GENERATED: docs/index.md -->

MLFCS builds primitive-cell force constants from ASE structures and forces. Its public workflow has three explicit objects:

1. `PrimitiveCell` and `build_cluster_space` define the primitive motif, symmetry, interaction cutoffs, and parameter space.
2. `Supercell` and `ClusterMap` connect that primitive model to one explicit supercell.
3. `FiniteDifference` reconstructs one force-constant order, while `FitSystem` builds and solves a joint force-only fit.

All structures use ASE. Lengths are in Å, energies in eV, and forces in eV/Å. The resulting order-`n` force constants have units eV/Åⁿ.

## Start with a workflow

- [Finite differences](docs/finite-difference-api.md): generate ordered displaced structures, evaluate an ASE calculator, and reconstruct one order.
- [Force fitting](docs/fitting-api.md): stream ASE structures with stored forces into a reusable fit system and solve multiple orders together.
- [Q&A](docs/Q&A.md): answers about force storage, ordering, identifiability, solvers, and common choices.

## Minimal model setup

```python
import numpy as np
from ase.build import bulk
from mlfcs import PrimitiveCell, Supercell, ClusterMap, build_cluster_space

primitive_atoms = bulk("Al", "fcc", a=4.05)
primitive = PrimitiveCell.from_atoms(primitive_atoms, symprec=1e-5)
space = build_cluster_space(
    primitive,
    cutoffs={2: 4.0, 3: 3.0},
    max_body_orders={2: 2, 3: 3},
)

supercell_atoms = primitive_atoms.repeat((3, 3, 3))
supercell = Supercell.from_atoms(
    primitive, supercell_atoms, matrix=np.diag([3, 3, 3])
)
mapping = ClusterMap.build(space, supercell)
```

The supercell is explicit input data; MLFCS does not silently choose or enlarge it. Check that it identifies the requested model with `mapping.rank_info()` before generating expensive forces. The examples in the API pages show the finite-difference and fitting paths separately.

## Scope

Each `FiniteDifference` object handles one order. A `FitSystem` can handle all orders in a single `ClusterSpace`. Force constants are primitive-cell objects; supercell realizations and external file formats are derived operations.

<!-- END GENERATED: docs/index.md -->

## Documentation

Read the [English documentation](https://gtiders.github.io/mlfcs/) or the [Chinese documentation](https://gtiders.github.io/mlfcs/zh/). The source pages live at [docs](docs/) for English and [docs/zh](docs/zh/) for Chinese.

## Citation

If MLFCS contributes to published work, cite the software metadata in [CITATION.cff](CITATION.cff).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [issue tracker](https://github.com/gtiders/mlfcs/issues).

## License

MLFCS is distributed under the [GNU General Public License v3.0 or later](LICENSE).
