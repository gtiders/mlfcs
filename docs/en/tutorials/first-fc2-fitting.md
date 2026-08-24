---
title: Force-only force-constant fitting
audience:
  - user
status: stable
code_verified: 4.0.0a5
---

# Force-only force-constant fitting

`mlfcs.fitting` estimates consecutive symmetry-reduced orders jointly from externally sampled ASE
structures. It is independent of the finite-difference `sow()` / `reap()` reconstruction path,
but both routes return the same `ForceConstants` type and use the same output writers.

Each training structure must use the reference-supercell atom order and provide forces through an
ASE calculator result or a `forces` array. Energies are not used. The force model is linear in the
irreducible force-constant parameters. Taylor displacement features are the default;
covariance-orthogonalized Wick features are an explicit optional representation for joint
high-order fits.

`FittingResult.fitting_parameters` is expressed in the selected basis, while
`FittingResult.force_constants` is always an ordinary Taylor artifact. With
`fitting_basis="taylor"`, lowering is the identity. With `fitting_basis="wick"`, the conversion
within the exported FC2--FCn range begins as

```text
Phi_T[m] = Phi_W[m] - 1/2 Phi_W[m+2]:Sigma
           + 1/8 Phi_W[m+4]:Sigma:Sigma - ...
```

This is a polynomial change of coordinates, not another fit. External formats never interpret
Wick parameters or covariance.

Odd Wick orders also contract to a Taylor FC1 (constant-force) term. FC1 has no standard phonon or
ShengBTE output role and is not included in `ForceConstants`; its maximum component and net force
are reported explicitly. Therefore the exported FC2--FCn tensors are the exact Taylor derivatives
in that order range, but they do not by themselves reproduce a non-zero omitted FC1.

Spatial and body-order supports are checked under every same-parity Wick contraction. A missing
target cluster is accepted only when its site-symmetry tensor space has zero dimension and the
fully image-aggregated contraction is zero within a scale-aware numerical tolerance. A non-zero
contraction into a symmetry-forbidden cluster is treated as a covariance, periodic-image, or
aggregation error. A symmetry-allowed missing cluster is instead reported as a genuine support
closure error; MLFCS never silently drops either contribution.

A non-zero reported FC1 is useful diagnostic information, not by itself evidence of an incorrect
fit. It can arise when the reference structure is not the statistical center or stationary point
best represented by the sampled data, from residual reference forces, finite-sample noise, an
asymmetric sampling distribution, truncation of the polynomial order, or spatial/body-order
cutoffs. Translational invariance constrains the net FC1 but does not require every atomic FC1 to
vanish. MLFCS therefore reports both the largest atomic component and the net value, and does not
silently constrain FC1 to zero. Imposing a zero Taylor FC1 would define a different constrained
regression problem and may be added only as an explicit option after separate validation.

Stationarity must be assessed from the reported Taylor FC1 after the Wick-to-Taylor conversion,
not directly from the solver's internal first-order Wick coefficient. Higher odd Wick orders
contract into Taylor FC1, so these quantities generally differ. At fixed lattice vectors, images
of each primitive site may be aggregated and an indicative local Newton correction
`Delta u = -Phi2^+ Phi1` may be estimated after removing the three rigid-translation zero modes.
This displacement is a reference-stationarity and data-quality diagnostic, and is credible as a
local suggestion only when it is much smaller than the sampled displacement range. It must not
relax equivalent supercell images independently, cannot determine lattice constants or cell
shape, and does not replace a first-principles relaxation with stress; moving the reference
normally requires new forces and a new fit.

Translational constraints commute with covariance contractions and remain part of the fit.
Born-Huang and Huang conditions are deliberately not fit-time Wick constraints: constraining
final Taylor FC2 in a joint FC2/FC4 Wick fit would modify FC4. Apply the explicit FC2-only
postprocessor to `result.force_constants` after fitting; it leaves all higher orders unchanged.

```python
from ase.io import read
from mlfcs.fitting import ForceConstantFitter

fitter = ForceConstantFitter(
    primitive=read("POSCAR"),
    reference=read("reference.xyz"),
    orders=(2, 3, 4),
    fitting_basis="wick",
    cutoffs={2: 8.0, 3: 12 * 0.529177210903, 4: 8 * 0.529177210903},
    max_body_orders={2: 2, 3: 3, 4: 3},
)
result = fitter.fit(
    read("train.xyz", index=":"),
    batch_size=4,
    validation_split=0.1,
    tolerance=1e-7,
    max_iterations=10_000,
    acoustic_sum_rule=True,
    allow_unconverged=False,
)
write_force_constants(result.force_constants, "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
write_force_constants(result.force_constants, "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
write_force_constants(result.force_constants, "FORCE_CONSTANTS_4TH", format="shengbte", order=4)
```

MLFCS does not freeze externally supplied low-order IFCs during a higher-order fit. All requested
orders are determined jointly in one Wick parameter space, so same-parity contractions, symmetry,
and equality constraints remain internally consistent.

`max_body_orders` optionally limits the number of distinct atomic sites in a cluster at each
order. For example, `(0, 0, 1, 1)` is a two-body fourth-order cluster. Omitting an order or using
`None` retains all body orders up to that force-constant order. The same definition is available
as `max_body_order` on `FiniteDifferenceCalculation`, so fitting and finite differences use an
identical interaction space.

The reference structure defines zero displacement. If it carries forces, they are treated as
residual reference forces and subtracted from every training target; without reference forces they
are assumed to be zero. Snapshot displacements and forces are otherwise left unchanged. In
particular, MLFCS does not silently remove a rigid translation or a snapshot's net force. The
maximum center-of-mass displacement, reference force, and snapshot net force are diagnostics so
that inconsistent input is visible without changing user data.

The fitter streams each design batch into the sufficient statistics
`A.T @ A` and `A.T @ F`
without storing the full snapshot-dependent design matrix. Orbit tensors are grouped automatically
by their exact image count and independent-parameter dimension; these internal buckets are not API
settings. Periodic geometry stores only atomic indices: Cartesian component tuples are generated
inside the JAX kernel rather than pre-expanded for every orbit, image, translation, and tensor
component. Each kernel returns only its local parameter columns, and large covariance and orbit
arrays are runtime arguments rather than captured XLA constants.

For unregularized fitting, hard constraints are parameterized before design accumulation. A
block-sparse map `Z` is constructed from constraint-connected components by pivoted QR, with
`theta = Z q`; the fitter accumulates `(A Z).T @ (A Z)` directly. Thus Gram storage and solving
scale with the constrained degrees of freedom, not the original irreducible parameter count. The
map itself remains sparse; a dense global null-space matrix is never formed. A `PreparedDesignProgram` packs orbit tiles, uploads static buffers, and caches JIT
callables once per fit; the same program is reused for training, validation, and diagnostics. On
CPU, JAX builds physical design tiles and OpenBLAS/SciPy perform sparse reduction and Gram
accumulation. On a JAX GPU backend, physical design, bounded sparse null-space reduction, and
Gram accumulation remain device-resident; only final sufficient statistics are transferred back.
Per-parameter exact column-norm preconditioning is obtained from the Gram diagonal. `batch_size`
is limited to 1--4 and controls only how many structures contribute to a design batch.

`fit(..., cache_directory="path")` is a stable public recovery-cache API. MLFCS fingerprints
the displacement, force, covariance, and parameterization inputs, then stores the completed Gram
statistics below `path/gram-<fingerprint>/`. A subsequent identical fit reuses those statistics;
changed inputs always select a different cache entry. The resulting `FittingResult.cache_directory`
reports the exact entry used (or `None` when caching was inactive).

Set `MLFCS_JAX_TRANSFER_GUARD=log` or `disallow` during development to audit unintended implicit
JAX transfers. The default is inert. When rotational constraints mix orders, order-resolved force
RMS is evaluated in one shared post-fit feature pass rather than one pass per order.

The default `regularization=None` solves the strictly constrained unregularized Gram problem.
`regularization="scaled_group_lasso"` reuses the same Gram statistics and applies an ADMM
concomitant-noise group penalty. One group is one complete symmetry-irreducible cluster orbit, so
an interaction orbit is retained or suppressed as a unit rather than selecting arbitrary tensor
components. Group thresholds account for orbit dimension, and the residual noise scale and
penalty magnitude are estimated during optimization; no user penalty or cross-validation is
required. ASR and optional rotational identities remain hard equality constraints in both modes.

For the default unregularized solve, equality constraints use the explicit block-sparse
parameterization above. The orbit-group LASSO path retains an implicit null-space projector based
on a rank-revealing pseudoinverse of `C @ C.T`, because a general null-space transformation would
destroy its orbit-local penalty groups. In both cases iteration remains in `null(C)` instead of
solving an indefinite KKT system and repairing its result afterward. `max_iterations` is a safety
limit: a zero solver status means the projected-gradient tolerance was reached, while a positive
status means that limit was reached without convergence. An unconverged solve raises by default
and cannot produce writable force constants. `allow_unconverged=True` is an explicit diagnostic
escape hatch and prints a warning in the result log.

The primary accuracy metric is defined directly from the reference and predicted forces:

```text
relative force error = ||F_reference - F_model||₂ / ||F_reference||₂
```

It is printed as a percentage together with force RMSE in eV/angstrom, validation error,
order-resolved force-contribution RMS, projected normal residual, and constraint drift.

For physical FC2 Born-Huang/Huang correction, call
`enforce_rotational_sum_rules(result.force_constants, ...)`. Its strict default is
`strength=1.0`; see [sum rules].
