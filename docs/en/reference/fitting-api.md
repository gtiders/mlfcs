---
title: Fitting API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# Fitting API

`ForceConstantFitter` uses Taylor coordinates exclusively. Gram construction is an explicit,
independent step so statistics can be saved and reused without retaining snapshots or a compiled
design plan.

```python
fitter = ForceConstantFitter(
    primitive,
    reference,
    orders=(2, 3),
    cutoffs={2: 5.4, 3: 4.5},
    max_body_orders={2: 2, 3: 3},
    periodic_fc2_completion=False,
    symprec=1e-5,
)
gram = fitter.prepare_gram(structures)
gram.save("training-gram.npz")
result = fitter.fit(gram, acoustic_sum_rule=True)
```

`prepare_gram()` accepts one user-owned dataset and returns portable sufficient statistics. It
processes one snapshot at a time and takes its parallelism from the interaction orbits, so no
snapshot batch size is exposed; the compiled design kernel is sized by the fitting problem, not by
a batch. `GramStatistics.load()` restores them on any host. `fit()` only solves and reconstructs
Taylor IFCs; it does not split validation data or calculate test-set predictions. The Gram matrix is
always built in the complete physical coordinate space and contains no ASR policy, so the same object
can be fitted both with and without ASR.


`FittingResult` contains the fitted force constants, Taylor parameters, Gram statistics, training
errors derived from the Gram quadratic form, solver state, and ASR projection diagnostics. The raw
least-squares parameters are retained as `unprojected_parameters`; `fitting_parameters` contains the
optional order-local Euclidean ASR projection. Force evaluation is owned by `MLFCSCalculator`.

Saved Gram statistics contain a physical-design identity. Loading, merging, or fitting statistics
from a different structure, order set, cutoff, or orbit parameterization is rejected explicitly.

## What a parameter means

A fitted parameter is a coefficient of its orbit's orthonormal Cartesian basis $Q$, so the parameter
vector has one coordinate meaning: the representative tensor is $Q\theta$, and an orbit's parameter
count equals the dimension of its invariant subspace. The exact integer coefficients
$c = R^{-1}\theta$ describe the same tensor in the lattice frame of the reduced cell and exist for
provenance and cross-representation comparison.

The component rows a finite-difference plan observes, the observation matrix and its condition number
travel with each orbit (`observation_rows`, `observation_matrix`, `observation_condition`). They decide
which components have to be measured and how well the parameters can be recovered from them; a
parameter is not an observation. See [Symmetry and orbits](../theory/symmetry-and-orbits.md).
