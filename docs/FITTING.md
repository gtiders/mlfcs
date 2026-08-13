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

Translational constraints commute with the covariance contractions and therefore remain valid
under this conversion. Rotational constraints couple adjacent Taylor orders and cannot be applied
directly to Wick coefficients. When `rotational_invariance=2` or `3`, MLFCS constructs the Taylor
constraint matrix `C_T`, the covariance-dependent Wick-to-Taylor map `T(Sigma)`, and solves with

```text
C_W = C_T @ T(Sigma).
```

The same `T(Sigma)` produces the exported Taylor IFCs. Thus the constrained fit and the output
use one definition, rather than projecting a completed result afterward. Results produced by an
earlier development implementation that enabled rotational fitting before this mapping was added
must be recomputed; ASR-only and `rotational_invariance=0` results are unaffected.

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

The fitter has one solver path. It streams each design batch into the sufficient statistics
`A.T @ A` and `A.T @ F`
without storing the full snapshot-dependent design matrix. Orbit tensors are grouped automatically
by their exact image count and independent-parameter dimension; these internal buckets are not API
settings. On CPU, JAX constructs each design batch and mature OpenBLAS/SciPy routines accumulate
and solve the Gram system. When a JAX GPU backend is active, design construction and Gram
accumulation remain on the GPU and only the completed Gram matrix is transferred once.
Per-parameter exact column-norm preconditioning is obtained from the Gram diagonal. `batch_size`
is limited to 1--4 and controls only how many structures contribute to a design batch.

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
order-resolved force-contribution RMS, projected normal residual, and constraint drift.

`rotational_invariance=2` follows ALAMODE `ICONST=2`: Cartesian adjacent-order rotational
constraints are imposed while the maximum-order/next-order boundary is omitted.
`rotational_invariance=3` additionally assumes the unrepresented next order is zero and imposes
the maximum-order boundary. The latter can overconstrain a truncated expansion and is never the
default. Fractional coordinates are not used for rigid rotations because a non-orthogonal cell
requires the Cartesian metric.
