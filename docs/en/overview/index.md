---
title: Getting started
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Getting started

MLFCS exposes Python APIs rather than a command-line interface. A calculation has three explicit
stages: construct a calculation, generate structures, and provide forces.

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from mlfcs import FiniteDifferenceCalculation
from mlfcs import build_supercell

primitive = bulk("Al", "fcc", a=4.05)
reference_supercell = build_supercell(primitive, (2, 2, 2))
calculation = FiniteDifferenceCalculation(
    primitive,
    reference=reference_supercell,
    order=2,
)
result = calculation.run(EMT())
write_force_constants(result, "fc2.h5", format="hdf5")
```

For expensive calculators, use [`sow()` and `reap()`](../tutorials/external-calculator.md)
and store a manifest alongside the returned files. See [installation](installation.md),
[the first FC2](../tutorials/first-fc2-finite-difference.md), and structure conventions.
