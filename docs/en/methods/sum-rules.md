# Sum rules

MLFCS applies the acoustic sum rule (ASR) in each finite-difference or fitting
calculation when `acoustic_sum_rule=True`, which is the default. It is an
order-local constraint in the orbit parameter space.

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
print(constrained.diagnostics)
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
