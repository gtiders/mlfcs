---
title: Exceptions
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# Exceptions and troubleshooting

The public workflows mostly raise standard Python exception types, with the rejected condition
named in the message. MLFCS never downgrades a failure to a warning.

| Exception | Usual cause | Check first |
|---|---|---|
| `TypeError` | Not an ASE `Atoms`/`Calculator`/`ForceConstants` | Object type and import origin |
| `ValueError` | Parameter range, structure relation, shape, order or format is invalid | The full message, the reference and the units |
| `KeyError` | A requested IFC order or temperature does not exist | `force_constants.orders`, the temperature schedule |
| `RuntimeError` | A numerical rank, normal form or internal iteration guarantee failed | The log of the same call and the rank information |
| `MemoryError` | A dense target tensor exceeds the available memory | Keep HDF5 sparse, avoid high-order materialization |
| `AlamodeMirrorImageError` | The target supercell cannot express the ALAMODE 27-image encoding | Change the target supercell or the output format |
| `SymmetryViolationError` | Force constants do not satisfy the space group of their own primitive cell | The operation index, q label and residual in the message; fix the force constants or relax `symmetry_tolerance` explicitly |

Fitting, finite difference and harmonic sampling all treat the reference atom order as
authoritative. If the input only differs by a permutation, use `align_structures()` explicitly;
do not expect a computational API to reorder it. If the lattice or the primitive does not
correspond, the input structures have to be fixed.

When several primitive exact-$R$ interactions fold onto one finite observation in the reference
and the realization map becomes rank deficient, the construction refuses to continue. More
training frames cannot repair the representation kernel; a larger reference or a shorter cutoff
is required.

Fitting rejects unconverged solves by default; only `allow_unconverged=True` returns the last
parameters with a warning. SCPH and SSCHA return the last iterate with `converged=False` when
they reach the iteration budget, so a caller must inspect the history and not merely the fact
that a file was written.

The reciprocal paths never average a symmetry error away. These situations raise, and the
message names where the failure comes from:

- the primitive symmetry cannot be determined (spglib returns nothing): the message reports the
  atom count, the cell and `symprec`;
- an operation does not keep the current grid, or the label action is not closed: the message
  reports the operation index and the label;
- a site permutation maps atoms of different mass onto each other: the message reports the
  operation and the site;
- the dynamical matrix is not covariant on the little group of a representative: the message
  reports the operation, the label, the residual, the matrix scale and the relative tolerance;
- an expanded matrix is not Hermitian: the message reports the label, the operation and the
  residual;
- the eigenbasis the sampler expanded cannot rebuild that member's dynamical matrix: the message
  reports the label and the residual.

`symprec` and `symmetry_tolerance` are two different tolerances: the first is the geometric
tolerance that identifies the structure, the second is the **relative** physical tolerance the
force constants have to satisfy (relative to the largest dynamical-matrix element of that grid).
Both are explicit arguments of the public constructors and are recorded in the results and the
metadata; only `symmetry_tolerance=None` switches the check off, and it is never off by default.
