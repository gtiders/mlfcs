---
title: External calculator workflow
audience:
  - user
status: stable
code_verified: 4.0.0a6
examples:
  - examples/finite-difference/Si/harmonic
---

# External calculator workflow

## Goal

Generate an ordered displacement plan, evaluate every structure with VASP or another external program, and reconstruct force constants without hidden atom reordering.

## Setup and sow

~~~python
from pathlib import Path
from ase.io import read, write
from mlfcs import FiniteDifferenceCalculation

calculation = FiniteDifferenceCalculation(
    read("primitive.vasp"),
    reference=read("reference.vasp"),
    order=3,
    cutoff=-5,
    displacement=0.01,
)
structures = calculation.sow()
for index, atoms in enumerate(structures):
    directory = Path("calculations") / f"{index:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "POSCAR", atoms, format="vasp")
~~~

Store the zero-based configuration index beside every submitted job. MLFCS does not create VASP inputs other than structures and does not submit jobs.

## Collect and reap

~~~python
import numpy as np
from ase.io import read
from mlfcs import write_force_constants

forces = np.asarray([
    read(f"calculations/{index:05d}/vasprun.xml", index=-1).get_forces()
    for index in range(len(structures))
])
fc3 = calculation.reap(forces, acoustic_sum_rule=True)
write_force_constants(fc3, "mlfcs.h5", format="hdf5")
~~~

The force at `forces[i]` must belong to `structures[i]`. For out-of-order completion, pass the supported configuration-ID mapping rather than guessing a sorted filename order.

## Results and next steps

Keep the structure manifest, calculator inputs, and raw outputs with the calculation provenance. The Si harmonic case demonstrates this separation using archived outputs, a force-collection script, a reconstruction script, and an independent plotting script.
