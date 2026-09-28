# SSCHA

`SSCHA` performs stochastic self-consistent harmonic force matching at a fixed cell and centroid. It samples paired displacements from a trial harmonic FC2, obtains fresh forces from an ASE calculator, refits FC2, applies ASR to each fitted update, and iterates the trial model.

## Construct `SSCHA`

```python
from mlfcs.reciprocal import SSCHA

solver = SSCHA(
    mapping,
    mapping.supercell.matrix,
    initial=fc2,
    statistics="quantum",
    seed=7,
)
```

| Parameter | Meaning |
| --- | --- |
| `mapping` | `ClusterMap` for an FC2-only `ClusterSpace`; its supercell must structurally identify FC2. |
| `mesh` | `QGrid`, a length-three diagonal mesh, or a 3×3 integer matrix. Its matrix must equal the declared supercell matrix. |
| `initial` | Optional FC2-only `ForceConstants` on the mapping's cluster space and with matching masses. If omitted, SSCHA creates an initial trial from sampled Cartesian displacements. |
| `statistics` | `"quantum"` (default) or `"classical"` modal statistics. Classical sampling requires positive temperature. |
| `seed` | Optional integer random seed in `[0, 2**64)`. If omitted, a seed is generated and recorded in results. |
| `bootstrap_displacement` | Optional positive Cartesian displacement scale in Å used to construct a trial when the start is absent or unstable. `None` leaves bootstrap disabled for an unstable supplied trial. |

The solver retains `mapping`, the irreducible `stars`, `initial`, `statistics`, `seed`, and `bootstrap_displacement`. The mesh is required to match the declared supercell, so the reciprocal sampling and the real-space force data refer to the same periodic cell.

## Run one temperature

```python
result = solver.run(
    300.0,
    calculator,
    pairs=128,
    max_pairs=512,
    max_iterations=20,
    tol_thz=0.01,
    mixing=0.5,
)
print(result.status, result.fc2.orders, len(result.history))
```

`run(temperature, calculator, *, pairs=128, max_pairs=None, max_iterations=20, tol_thz=0.01, mixing=0.5, max_backtracks=8, energy=False, start=None)` accepts:

| Parameter | Meaning |
| --- | --- |
| `temperature` | Finite nonnegative temperature in K. At zero temperature, use quantum statistics. |
| `calculator` | ASE `Calculator` used for freshly evaluated forces; energy is also requested when `energy=True`. |
| `pairs` | Initial number of independent displacement pairs; must be at least 2. Each pair evaluates positive and negative displacements. |
| `max_pairs` | Maximum sample pairs. `None` sets it equal to `pairs`; uncertainty-driven sampling can increase the count up to this value. |
| `max_iterations` | Maximum self-consistent updates. |
| `tol_thz` | Both the frequency-change and sampling-uncertainty target, in THz. |
| `mixing` | Initial accepted FC2 update fraction in `(0, 1]`. |
| `max_backtracks` | Number of halvings tried if an FC2 update produces an unstable trial. |
| `energy` | Whether to evaluate potential energies and estimate the anharmonic free energy. Default `False`. |
| `start` | Optional FC2-only model on this mapping and with matching masses. Overrides the constructor's `initial` for this run. |

If no trial is supplied, the algorithm bootstraps one from paired random Cartesian displacements. For an unstable trial, a positive `bootstrap_displacement` enables the same recovery path. Sampling does not silently replace negative curvatures by positive ones: unstable trials are reported and candidate updates are backtracked until stable or the configured limit is reached.

`SSCHAResult` contains `temperature`, `fc2`, `history`, `status`, `n_qpoints`, `n_irreducible`, `mapping_fingerprint`, `seed`, `message`, and `bootstrap_displacement`; `converged` is derived from `status`. Status values are `converged`, `max_iterations`, `insufficient_samples`, and `unstable`.

Each `SSCHAStep` records the pair count, frequency change and uncertainty, minimum mode, accepted mixing, fit RMSE, ASR residual, mean force, and optional free-energy estimate and error.

## Run a temperature series

```python
results = solver.run_many(
    [300, 400, 500],
    calculator,
    pairs=128,
    max_pairs=512,
)
```

`run_many(temperatures, calculator, **kwargs)` requires a strictly increasing sequence of finite nonnegative temperatures. It executes from high to low temperature to warm-start each lower-temperature run from the preceding FC2, then returns results in ascending input order. A nonconverged result stops the schedule before lower temperatures and raises `SSCHAContinuationError`; completed results and the stopping result are available on the exception.

## Masses and symmetry

SSCHA starts from `ClusterMap`; masses come from `mapping.space.primitive.masses` and are propagated to sampled ASE structures. The current constructor has no independent mass override. Set the desired masses on the ASE primitive `Atoms` before constructing `ClusterSpace` and its mapping. Harmonic mass weighting and reciprocal star reduction must remain compatible: the harmonic path rejects masses that are not invariant under the model's primitive symmetry.
