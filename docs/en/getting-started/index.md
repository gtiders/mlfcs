# Getting started

MLFCS exposes Python APIs rather than a command-line interface. A calculation has three explicit
stages: construct a calculation, generate structures, and provide forces.

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from mlfcs import ForceConstantCalculation

primitive = bulk("Al", "fcc", a=4.05)
calculation = ForceConstantCalculation(
    primitive,
    order=2,
    supercell=(2, 2, 2),
    calculator=EMT(),
)
result = calculation.run()
result.write("fc2.h5", format="hdf5")
```

For expensive calculators, use [`sow()` and `reap()`](../workflows/external-calculators.md)
and store a manifest alongside the returned files. See [installation](installation.md),
[the first FC2](first-fc2.md), and [structure conventions](structures.md).
