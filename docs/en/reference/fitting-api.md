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
independent step so statistics can be saved and reused without retaining snapshots or a JAX
operator.

```python
fitter = ForceConstantFitter(
    primitive,
    reference,
    orders=(2, 3),
    cutoffs={2: 5.4, 3: 4.5},
    max_body_orders={2: 2, 3: 3},
    periodic_fc2_completion=False,
    symprec=1e-5,
    jax_platform="auto",
)
gram = fitter.prepare_gram(structures, batch_size=1, acoustic_sum_rule=True)
gram.save("training-gram.npz")
result = fitter.fit(gram, acoustic_sum_rule=True)
```

`prepare_gram()` accepts one user-owned dataset and returns portable sufficient statistics.
`GramStatistics.load()` restores them on CPU or GPU hosts. `fit()` only solves and reconstructs
Taylor IFCs; it does not split validation data or calculate test-set predictions.

`FittingResult` contains the fitted force constants, Taylor parameters, Gram statistics, training
error derived from the Gram quadratic form, solver state, constraint residuals, regularization
state, and optional periodic FC2 completion. Force evaluation is owned by `MLFCSCalculator`.

Periodic completion requires FC2, strict ASR, and unregularized least squares. The transferable
exact-$R$ FC2 remains in `force_constants.sparse[2]`; the source-owned finite Hessian remains in
`force_constants.periodic_fc2_completion`.
