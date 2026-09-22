---
title: Units and Parameters
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# Units and parameters

MLFCS uses ASE units: energy in eV, Cartesian length in angstrom, force in eV/angstrom, and an
order-$n$ force constant in eV/angstrom$^n$.

## Structure precision

| Parameter | Unit | Meaning |
|---|---:|---|
| `symprec` | angstrom | Primitive symmetry, primitive/reference lattice relation, atom mapping, and fixed-cell identity |

`symprec` asks one physical question: how far apart two periodic Cartesian geometries may be while
still representing the same object. Dimensionless matrix entries, fractional-coordinate errors,
and angles are not compared directly with it. The core does not expose separate
`mapping_tolerance`, `cell_tolerance`, or `position_tolerance` parameters.

spglib's `angle_tolerance` remains on its automatic policy and is not a second MLFCS length
precision. The required `tolerance` of `mlfcs.tools.structure_alignment` is a separate external
data-import policy and is never consumed by the computational core.

Every computational entry that needs a finite cell requires an explicit `reference` without a
default. `mlfcs.tools.supercell.build_supercell` is optional convenience code; the core does not
derive a supercell from a cutoff.

## Other tolerances

Solver convergence tolerances, frequency cutoffs, displacement amplitudes, and interaction cutoffs
do not identify crystal geometry and therefore do not share `symprec`. Their units and acceptance
rules are documented by the API that owns them.
