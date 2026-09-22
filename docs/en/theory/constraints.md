---
title: Sum rules
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# Sum rules

MLFCS applies the acoustic sum rule (ASR) in each finite-difference or fitting
calculation when `acoustic_sum_rule=True`, which is the default. Both paths first recover
unconstrained physical orbit coefficients $\theta_0$ and then use the same order-local Euclidean
projection onto $A\theta=0$. ASR is therefore a physical postprocessing operation, not a reduced
fitting coordinate system or a constrained least-squares estimator.

This distinction is deliberate. A fitting Gram matrix describes only the force observations and is
reusable with ASR enabled or disabled. Projection can increase the training residual because it
chooses the closest ASR-feasible parameter vector to the unconstrained solution, rather than the
ASR-feasible vector that minimizes the training loss. The result reports both values.

Born-Huang and Huang conditions have different semantics. They are physical
FC2-only postprocessing conditions applied after force constants have been
constructed or read from native HDF5:

```python
from mlfcs import read_hdf5

result = read_hdf5("mlfcs.h5")
constrained = enforce_rotational_sum_rules(result, 
    born_huang=True,
    huang=True,
)
fc2 = constrained.force_constants
print(constrained)
```

`strength=1.0` is the default and denotes the strict retained-rank projection.
Values between zero and one scale only the Born-Huang/Huang correction; ASR is
always reimposed exactly. `tolerance` is a dimensionless spectral cutoff after
pair distances are normalized by the median nonzero nearest-image distance.

The projector uses the verified `StructureRelation` and lattice-labelled sparse
FC2. Tied nearest images use equal weights: Born-Huang uses their mean vector,
and Huang uses their mean dyadic. It returns a new result, preserves the raw
result, and leaves FC3, FC4, and every other order unchanged.

Huang is the zero-stress condition. It is meaningful only for a stress-free
reference and is not a replacement for long-range electrostatics or NAC.
