---
title: First finite-difference FC2
audience:
  - beginner
status: stable
code_verified: 4.0.0a6
examples:
  - examples/finite-difference/Si/harmonic
---

# First finite-difference FC2

## Goal

Calculate a harmonic FC2 with a direct ASE calculator and save it in native sparse HDF5.

## Prerequisites

Install the project with `uv sync`. This small example uses ASE's EMT calculator and requires no external electronic-structure program.

## Steps

~~~python
from ase.build import bulk
from ase.calculators.emt import EMT
from mlfcs import FiniteDifferenceCalculation, write_force_constants
from mlfcs.tools.supercell import build_supercell

primitive = bulk("Al", "fcc", a=4.05)
reference = build_supercell(primitive, (2, 2, 2))

calculation = FiniteDifferenceCalculation(
    primitive,
    reference=reference,
    order=2,
    cutoff=7.7237404951,
    displacement=0.01,
)
force_constants = calculation.run(EMT())
write_force_constants(force_constants, "mlfcs.h5", format="hdf5")
~~~

## Results and interpretation

`mlfcs.h5` contains native HDF5 v4 sparse exact-$R$ FC2. Use an explicit writer to create dense phonopy output when a downstream workflow requires it.

## Common problems

The primitive and reference must describe one exact integer-supercell relation. The cutoff belongs to the primitive model and is always explicit: a positive number is a distance in angstrom and a negative integer is a neighbour shell, so the model never depends on how large the reference that observes it happens to be. Here the radius is the boundary the $4\times4\times4$ reference of the Si case resolves; a real calculator may need a larger reference and a radius justified by convergence instead.

## Next steps

The repository's Si harmonic case reconstructs FC2 from archived VASP outputs and plots the resulting phonon bands.
