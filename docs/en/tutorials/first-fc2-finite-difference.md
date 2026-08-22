---
title: First finite-difference FC2
audience:
  - beginner
status: stable
code_verified: 4.0.0a4
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
from mlfcs import ForceConstantCalculation, build_supercell, write_force_constants

primitive = bulk("Al", "fcc", a=4.05)
reference = build_supercell(primitive, (2, 2, 2))

calculation = ForceConstantCalculation(
    primitive,
    reference=reference,
    order=2,
    cutoff=None,
    displacement=0.01,
)
force_constants = calculation.run(EMT())
write_force_constants(force_constants, "mlfcs.h5", format="hdf5")
~~~

## Results and interpretation

`mlfcs.h5` contains native HDF5 v3 sparse exact-$R$ FC2. Use an [explicit writer](../how-to/read-and-write-ifcs.md) to create dense phonopy output when a downstream workflow requires it.

## Common problems

The primitive and reference must describe one exact integer-supercell relation. A real calculator may require a larger reference and a cutoff justified by convergence rather than this minimal demonstration.

## Next steps

The repository's [Si harmonic case](../examples/si-finite-difference.md) reconstructs FC2 from archived VASP outputs and plots the resulting phonon bands.
