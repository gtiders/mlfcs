---
title: Structures API
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# Structures, supercells, and alignment

## `mlfcs.tools.supercell.build_supercell`

```python
from mlfcs.tools.supercell import build_supercell

build_supercell(
    primitive: Atoms,
    supercell_matrix: object,
    *,
    symprec: float = 1e-5,
) -> Atoms
```

This optional leaf utility creates a periodic ASE `Atoms` in primitive-site-major ordering using
the project's NumPy/ASE implementation; it has no phonopy runtime dependency.
`supercell_matrix` must be an integer triple or a nonsingular integer $3\times3$ matrix;
floating representations such as `[[2.0, ...]]` are rejected because the matrix is a discrete
construction parameter. `symprec` is the Cartesian length used to remove duplicate generated
sites on the boundary of the surrounding frame.

The computational core never imports `mlfcs.tools` and never constructs or guesses a reference
supercell. A caller may use this convenience function, read a supercell from a file, or obtain one
from another program, then pass that explicit `reference` to the calculation.

## `mlfcs.tools.supercell.align_structures`

```python
from mlfcs.tools.supercell import align_structures

align_structures(
    reference: Atoms,
    atoms: Atoms,
    *,
    tolerance: float,
) -> tuple[Atoms, float]
```

This external-import utility explicitly reorders an independently produced MD or simulation frame
to the reference atom order and returns the maximum observed cell/atom residual. The caller must
supply the finite positive `tolerance`; it is not the core structure identity precision. Fitting
and finite-difference paths never call this function implicitly.

## `StructureRelation`

```python
StructureRelation.from_atoms(
    primitive: Atoms,
    reference: Atoms,
    *,
    symprec: float,
) -> StructureRelation
```

The relation verifies that the explicit reference is an integer supercell of the primitive. It
stores the integer `supercell_matrix`, the one-to-one `(primitive_site, translation)` labels,
`symprec`, and the measured `cell_residual` and `position_residual` in angstrom.

`symprec` is the only core length precision. The lattice residual is normalized per primitive
lattice coefficient, while atom matching uses exact minimum-image Cartesian distances and a global
per-species assignment. Both residuals must be strictly smaller than `symprec`. The removed
`tolerance` keyword has no compatibility alias.

`StructureRelation.displacement(atoms)` also uses the stored `symprec` to verify the fixed reference
cell. Atomic displacements themselves may be much larger: they are the physical quantity returned
by the method, not a structure-identity error.
