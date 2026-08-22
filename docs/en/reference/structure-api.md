---
title: Structure API
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Structure API

Document supercell construction, structure relations, explicit alignment, and periodic indexing.

~~~python
build_supercell(
    primitive: Atoms,
    supercell_matrix: object,
    *,
    symprec: float = 1e-5,
) -> Atoms

align_structures(
    reference: Atoms,
    atoms: Atoms,
    *,
    tolerance: float = 1e-5,
) -> tuple[Atoms, float]
~~~

`build_supercell` accepts a length-three repetition or integer $3\times3$ matrix and returns phonopy old-style atom ordering. `align_structures` is explicit; fitting and finite-difference workflows never silently reorder snapshots.
