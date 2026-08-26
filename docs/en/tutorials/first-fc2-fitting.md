---
title: First FC2 fit
audience:
  - user
status: stable
code_verified: 4.0.0a6
---

# First FC2 fit

Construct one Taylor interaction model, accumulate its Gram statistics, then solve it:

```python
from ase.io import read

from mlfcs import ForceConstantFitter, write_force_constants

primitive = read("primitive.vasp")
reference = read("supercell.vasp")
training = read("train.extxyz", index=":")

fitter = ForceConstantFitter(
    primitive,
    reference,
    orders=(2,),
    cutoffs={2: 5.4},
    max_body_orders={2: 2},
)
gram = fitter.prepare_gram(training, batch_size=1, acoustic_sum_rule=True)
gram.save("training-gram.npz")
result = fitter.fit(gram, acoustic_sum_rule=True)
write_force_constants(result.force_constants, "mlfcs.h5", format="hdf5")
```

The Gram archive contains $A^T A$, $A^T f$, $f^T f$, the equation count, and portable parameter
metadata. It can be loaded with `GramStatistics.load()` and fitted again without rebuilding design
features. Build a separate Gram object for validation or test data; `fit()` never predicts those
forces implicitly. Use `MLFCSCalculator` for explicit model-force evaluation.

All fitted and stored parameters are Taylor coefficients. ASR is imposed in the same physical
parameter space before reconstruction.
