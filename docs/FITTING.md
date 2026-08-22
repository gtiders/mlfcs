# Force-only force-constant fitting

`mlfcs.fitting` estimates consecutive symmetry-reduced orders jointly from externally sampled ASE
structures. It is independent of the finite-difference `sow()` / `reap()` reconstruction path,
but both routes return the same `ForceConstants` type and use the same output writers.

Each training structure must use the reference-supercell atom order and provide forces through an
ASE calculator result or a `forces` array. Energies are not used. The force model is linear in the
irreducible force-constant parameters and uses covariance-orthogonalized Wick displacement
features to reduce leakage between adjacent orders.

The fitted parameter vector is expressed in the Wick basis. Public `ForceConstants` output is
converted exactly to ordinary Taylor IFCs before serialization, because phonopy, ShengBTE, and
other force-constant formats interpret tensors as Taylor derivatives. For displacement
covariance `Sigma`, the conversion begins as

```text
Phi_T[m] = Phi_W[m] - 1/2 Phi_W[m+2]:Sigma
           + 1/8 Phi_W[m+4]:Sigma:Sigma - ...
```

This is a polynomial change of coordinates, not another fit. `FittingResult.parameters` remains
the fitted Wick parameter vector; `FittingResult.force_constants` contains Taylor IFCs ready for
common output formats.

```python
from ase.io import read
from mlfcs.fitting import ForceConstantFitter

fitter = ForceConstantFitter(
    primitive=read("POSCAR"),
    reference=read("reference.xyz"),
    supercell=(2, 2, 3),
    orders=(2, 3, 4),
    cutoffs={2: None, 3: 12 * 0.529177210903, 4: 8 * 0.529177210903},
)
result = fitter.fit(
    read("train.xyz", index=":"),
    solver="gram",
    batch_size=4,
    validation_split=0.1,
    tolerance=1e-7,
    max_iterations=10_000,
    acoustic_sum_rule=True,
    rotational_invariance=2,
)
result.force_constants.write("FORCE_CONSTANTS_2ND", format="phonopy", order=2)
result.force_constants.write("FORCE_CONSTANTS_3RD", format="shengbte", order=3)
result.force_constants.write("FORCE_CONSTANTS_4TH", format="shengbte", order=4)
```

The default solver is matrix-free LSMR. It evaluates `A @ x` and `A.T @ r` in bounded JAX batches
instead of materializing the force design matrix. Per-parameter column-norm preconditioning puts
features from all fitted orders on comparable numerical scales. `verbose=True` prints
column-estimation and
operator progress plus LSMR iteration diagnostics.

Set `solver="cached_lsmr"` when repeated matrix-free evaluations dominate runtime. MLFCS then
constructs the exact linear force-design matrix in JAX batches, stores it in an automatically
managed temporary disk mapping, and reuses it throughout LSMR. The cache path, storage precision,
and internal matrix blocks are intentionally not API options; the cache is deleted after fitting.
This backend trades temporary storage and operating-system page cache for speed. `batch_size` is
limited to 1--4 for both backends and controls only the number of structures processed together.

`solver="gram"` streams each design batch into the sufficient statistics `A.T @ A` and `A.T @ F`
without storing the full snapshot-dependent design matrix. Orbit tensors are grouped automatically
by their exact image count and independent-parameter dimension; these internal buckets are not API
settings. On CPU, JAX constructs each design batch and mature OpenBLAS/SciPy routines accumulate
and solve the Gram system. When a JAX GPU backend is active, design construction and Gram
accumulation remain on the GPU and only the completed Gram matrix is transferred once.

Equality constraints are enforced through an implicit null-space projector based on a
rank-revealing pseudoinverse of `C @ C.T`; projected conjugate gradient therefore remains in
`null(C)` instead of
solving an indefinite KKT system and repairing its result afterward. `max_iterations` is a safety
limit: a zero solver status means the projected-gradient tolerance was reached, while a positive
status means that limit was reached without convergence.

The primary accuracy metric follows ALAMODE:

```text
relative force error = ||F_reference - F_model||₂ / ||F_reference||₂
```

It is printed as a percentage together with force RMSE in eV/angstrom, validation error,
order-resolved force-contribution RMS, LSMR residuals, condition estimate, and constraint drift.

`rotational_invariance=2` follows ALAMODE `ICONST=2`: Cartesian adjacent-order rotational
constraints are imposed while the maximum-order/next-order boundary is omitted.
`rotational_invariance=3` additionally assumes the unrepresented next order is zero and imposes
the maximum-order boundary. The latter can overconstrain a truncated expansion and is never the
default. Fractional coordinates are not used for rigid rotations because a non-orthogonal cell
requires the Cartesian metric.
