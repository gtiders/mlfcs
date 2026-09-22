# Changelog

[中文](CHANGELOG_ZH.md)

All notable changes are documented here. Releases follow semantic versioning.

## Unreleased

- Consolidated optional structure utilities in `mlfcs.tools.supercell`: supercell construction now
  always uses the project implementation without importing phonopy, and `align_structures` moved
  into the same module without a compatibility shim.
- **Breaking:** independent Cartesian Gaussian perturbation moved from the root namespace and the
  removed `mlfcs.sampling` package to `mlfcs.tools.gaussian`. Reciprocal-space harmonic sampling
  remains under `mlfcs.reciprocal`; production packages do not import the leaf tools package.

### Changed

- **Breaking:** `StructureRelation.from_atoms(primitive, reference, tolerance=...)` is now
  `from_atoms(primitive, reference, symprec=...)`.  There is no alias: the old keyword raises
  `TypeError` naming it.  `symprec` is the single length precision of the primitive-supercell
  geometry, in angstrom, and it is recorded on the relation together with `cell_residual` and
  `position_residual`.  The lattice residual is expressed per primitive lattice coefficient, so
  one `symprec` means the same thing for a 1x1x1 cell and for a large repeat, and a dimensionless
  matrix difference is never compared with it.
- **Breaking:** `build_supercell` moved to `mlfcs.tools.supercell`.  `from mlfcs import
  build_supercell` and `from mlfcs.structure import build_supercell` no longer work and no
  forwarding alias is kept.  `mlfcs.tools` is a leaf package: it may import `mlfcs.structure`, and
  no core package imports it back.  Every computational entry point (`InteractionSpace`,
  `FiniteDifferenceCalculation`, `ForceConstantFitter`, `SSCHA`) keeps taking an explicit
  `reference` with no default; the core never builds a supercell for you.
- **Breaking:** `align_structures` moved to `mlfcs.tools.supercell` and its `tolerance`
  is now required.  It is an external-import policy for structures produced elsewhere, not a
  structure-identity threshold, and no core path calls it.
- `StructureRelation.displacement` validates fixed-cell training frames with the stored `symprec`
  instead of a hidden `1e-7`; real atomic displacements are returned whatever their size, while a
  varying cell is refused with its angstrom residual.
- `normalize_supercell_matrix` accepts only discrete input: an integer dtype or Python/NumPy
  integers.  A floating-point matrix is refused instead of being rounded with a `1e-10`
  comparison.
- **Breaking:** the native HDF5 schema is now v4. The writer records `symprec`, and the reader
  builds the canonical identity relation from it. Native v3 files are rejected rather than being
  interpreted under a different required-field contract.

### Added

- `mlfcs.reciprocal.symmetry` states the space-group representation of the mass-weighted
  displacement space once, `D(gq) = U_g(q) D(q) U_g(q)^dagger`, with `U_g(q)` built from the
  exact integer rotation and the primitive site permutation. A force-constant set that breaks
  its crystal symmetry is reported with the operation, the q label and the residual instead of
  being silently averaged into a symmetric one.
- `mlfcs.reciprocal.fourier` exposes the phase vector and tensor of every primitive-lattice
  term of an order-2 force-constant set, so the lattice gauge and the compact kernel of the
  samplers can be compared term by term.
- `mlfcs.reciprocal.modes` states the modal space once for SCPH and the harmonic sampler:
  `ModePolicy` (statistics, temperature, cutoff and an explicit `error`/`absolute`/`exclude`
  policy for a negative eigenvalue), `ModalCovariance`, `modal_eigenpairs`,
  `mass_weighted_translations`, `internal_mode_basis`, `gamma_acoustic_residual`,
  `require_finite` and `require_star_covariance_matrix`. Both consumers use the same Gamma
  internal subspace and the same weights, so neither can drift into its own convention.
- The frequency, covariance and sampling paths report their irreducible wedge:
  `harmonic_frequencies` returns a `HarmonicMeshResult` and `LoopSCPHResult` carries
  `irreducible_qpoints`, `irreducible_frequencies`, `weights` and the star decomposition,
  with `full_qpoints()` and `expand_frequencies()` for the explicit full mesh. The ambiguous
  `HarmonicSampler.qpoints` property is replaced by `irreducible_qpoints`, `weights` and
  `full_qpoints()`, and `SamplingState`/`SSCHAIteration` report both `n_qpoints` and
  `n_irreducible`.
- `symprec` and `symmetry_tolerance` are public arguments of `LoopSCPH`,
  `harmonic_frequencies`, `HarmonicSampler` and `SSCHA`, recorded in their results and in the
  effective force-constant metadata instead of being module constants. `LoopSCPH` and
  `harmonic_frequencies` also take an explicit `time_reversal`; `HarmonicSampler` does not
  expose it, because a real sampler only supports real force constants and real displacements
  and therefore always closes its `q/-q` pairing with time reversal, which its state records.
- `SymmetryViolationError` and an explicit `symmetry_tolerance` on `LoopSCPH`,
  `HarmonicSampler` and `harmonic_frequencies`: every reciprocal consumer now checks that the
  force constants are covariant on the little group of each star representative before it
  expands, reports the operation, label, residual and scale of a violation instead of
  averaging it away, and records the tolerance in its results and metadata.
- `scripts/benchmark_reciprocal_reduction.py` reports the reduction ratio, the
  diagonalization counts, the timings and the peak memory of the irreducible paths against a
  full-grid reference.

### Changed

- SCPH frequencies, the SCPH covariance and harmonic sampling now diagonalize only star
  representatives and expand to the full grid exactly, through the space-group
  representation and the positional gauge; expansion never re-enters an eigensolver, the
  covariance sum still runs over every full q point, workers are scheduled over
  representatives, and the SCPH stopping metric is the star-weighted full-grid RMS
  `sqrt(sum_s w_s ||w_s^n - w_s^{n-1}||^2 / (N_q N_b))`.
- **Breaking:** the three Gamma translations are removed from the modal space before
  `1 / omega^2` is formed, instead of being weighted like modes and hidden afterwards by
  `frequency_cutoff_thz`. The default-cutoff covariance of the hcp cell drops from `9.6e12` to
  `2.4e-3` and agrees with the cutoff-guarded path; a Gamma mode count changes by three.
- **Breaking:** `imaginary_tolerance` is removed from harmonic sampling and SSCHA. A negative
  eigenvalue is an imaginary mode without an arbitrary numerical boundary. SCPH defaults to
  `imaginary_modes="absolute"`, because an unstable trial lattice is normal in a self-consistent
  calculation; callers may still choose `exclude` or the diagnostic `error` policy explicitly.
- The covariance and Hermiticity gates refuse non-finite data before comparing, and even when
  `symmetry_tolerance` is `None`, where a NaN made every comparison false. A star-gate
  rejection whose cause is a broken acoustic sum rule now reports `||D(Gamma) B||` against the
  grid scale instead of blaming a star member.

### Fixed

- The crystal-symmetry gate is on by default, so a hand-built or heavily truncated model that
  does not satisfy its own space group is now rejected with the offending operation and label.
  `symmetry_tolerance=None` switches the check off explicitly for such fixtures; a structure
  whose symmetry spglib cannot determine now reports the atom count, cell and `symprec`.
- Harmonic sampling kept the full random degrees of freedom: every full q point draws its
  own coefficient, a `q/-q` pair draws on one side and takes the real part of that single
  complex amplitude, and a label with `q = -q + G` spans its real amplitude space. Two
  defects are fixed by that: the pair branch used to carry half of the physical variance on
  grids with a general `q/-q` pair, and a self-conjugate label used a single draw whose
  discarded imaginary part made the covariance depend on the arbitrary eigensolver phase.
  The sampled numbers change for those grids; the sampler-implied covariance now equals the
  exact Cartesian supercell covariance, and the mean-zero and second-moment checks against
  theory are kept.
- The covariance relation is evaluated on the *unreduced* rotated q label. Reducing modulo the
  grid first is a primitive reciprocal lattice translation, and the positional gauge of the
  dynamical matrix turns such a translation into the site-diagonal factor
  `diag(exp(2 pi i G . tau_a))`, so a reduced label made every grid-preserving operation look
  like a violation on any lattice with more than one atom per cell. `rotate_labels` and
  `rotate_label` now take `reduce=` and keep the reduced action for label sets such as stars
  and orbits.

## 4.0.0a6 — 2026-09-20

### Changed

- The public namespace imports every workflow eagerly: the lazy `__getattr__` loader for
  `ForceConstantFitter`, `LoopSCPH`, `SSCHA`, and `perturb_structures` is gone, so `import mlfcs`
  loads the fitting and finite-temperature stacks together.
- `prepare_gram()` no longer takes a `batch_size`: snapshots are streamed one at a time and the
  compiled design kernel draws its parallelism from interaction orbits, so no snapshot batching
  knob is exposed and the design matrix working set is one structure's.
- Force-design construction and Gram statistics are built by compiled Numba kernels instead of
  JAX. The GPU path, the `jax_platform` fitting argument, and the `jax` runtime dependency are
  removed: `prepare_gram()` runs one compiled parallel design kernel per IFC order and accumulates
  the Gram matrix with OpenBLAS, so the fitting stack is portable and no longer materializes XLA
  tile buffers. Force-constant metadata no longer carries a `jax_platform` field.

### Fixed

- `PeriodicGeometry` now resolves the minimum image through ASE's Minkowski-reduction based search
  instead of `ase.geometry.find_mic`. `find_mic` skips the reduction whenever the folded vector is
  shorter than `0.5 * min(cell.lengths())`, and that bound is not the inradius of the Wigner-Seitz
  cell, so skewed cells received a non-minimum image. Minimum-image lengths, degenerate image sets,
  and cluster image selection now agree with the true minimum image for skewed and unimodularly
  transformed frames; the `mic()` calling convention is unchanged.

### Added

- `LatticeFrame` (`mlfcs.structure.lattice_frame`) records the exact integer change of frame from the
  user's primitive cell to a canonical Minkowski-reduced algebra cell: `source_cell`, `algebra_cell`,
  the unimodular `source_to_algebra`/`algebra_to_source`, the motif match, exact conversions of
  fractional coordinates, integer translations and rotations, and the single order-$n$ tensor map
  $K_n = (A^{\mathsf T})^{\otimes n}$. Equivalent unimodular inputs reduce to the same frame.
- `mlfcs.interactions.algebra.exact` certifies exact ranks modulo primes: a modular rank never exceeds
  the rank over $\mathbb Q$, so one full-rank prime settles it, and a deficient verdict is certified once
  the product of the distinct primes exceeds the Hadamard bound of the largest minors. The kernel comes
  from a Smith normal form decomposition, so it is the *saturated* integer kernel rather than columns
  divided by their greatest common divisors. `RankCertificateError` reports a certificate that the
  on-demand prime stream cannot settle, and `IntegerRangeError` reports a lattice integer that does not
  fit `int64` instead of letting a C extension raise an unrelated conversion error.

### Changed

- The orbit algebra is decided in the reduced lattice (scaled) frame, where spglib rotations are integer
  for every cell. The invariant kernel is the exact kernel of the stacked stabilizer constraints, its
  dimension is certified, and the returned integer basis is verified against the constraints; the
  floating point Gram matrix, its eigenvalue threshold and `normalize_pivot_basis` are gone.
- `PrimitiveInteractionOrbit` exposes `exact_lattice_basis` ($B_{\mathbb Z}$, integers), `cartesian_basis`
  ($Q$, orthonormal) and `coefficient_transform` ($R$, with $C = K_n B_{\mathbb Z} = QR$) instead of one
  ambiguous `basis`. The fitted parameters are the coefficients of $Q$, so parameter *values* change while
  orbit counts, invariant dimensions and delivered force constants do not.
- `pivots` became `observation_rows`, together with `observation_matrix` and `observation_condition`.
  The rows are the components a finite-difference plan observes, chosen to maximize the volume of the
  observation block, and `reconstruct_sparse` solves $Q_{\mathrm{obs}}\theta = y_{\mathrm{obs}}$
  explicitly instead of assuming that an observed component is a parameter.
- Realization identifiability ranks the exact integer realization matrix with the modular certificate:
  no coefficient filter, no rank tolerance and no float fallback on irrational frames.
- `TensorAction` carries the lattice rotation of every operation, so stabilizer deduplication,
  composition and inversion are exact integers; the 1e-12 rounded Cartesian signature is gone.
- Orbit keys are mapped back to the user's cell with `LatticeFrame.source_labels` (exact integers), so
  reference supercells, `sow`/`reap` plans and I/O keep addressing the same physical interactions.
- `ReferenceFrame` carries the calculation's `lattice_frame`, and
  `build_primitive_interaction_space` takes `frame=` in place of `symmetry=` and `tolerance`.
- `cutoff=None`, group LASSO/ADMM fitting and every other unrelated capability are untouched.

### Fixed

- A large unimodular shear no longer inflates the integer algebra at third and fourth order: the canonical
  algebra frame makes equivalent representations produce identical integer bases, observation rows and
  condition numbers, and identical finite-difference reconstructions.
- Documentation: the symmetry-and-orbits theory pages explain the lattice frame, the two orbit bases and
  the observation rows, and the documentation test now runs `scripts/check_docs.py`, so the repository
  math-delimiter rule and the bilingual mirror are enforced by the suite.


### Added

- Finite-difference plans carry a canonical, serializable identity. `DisplacementManifest`
  records the schema version, the primitive and reference structure fingerprints, order,
  cutoff, body order, symprec, displacement, derivative backend, stencil signs, the
  extrapolation description, the canonical orbits (representative, dimension, observation
  rows, integer exact basis, images), the sorted displacement keys and every configuration
  (id, key, atoms, directions, signs, step). Its `fingerprint` is a SHA-256 hash of one
  canonical JSON document with sorted keys and `float.hex()` encodings, so no `repr()`,
  object hash, memory layout or raw float bytes enter the value. `save`/`load` round-trip
  the manifest and recompute the fingerprint, rejecting an edited file.
- `sow()` returns a `DisplacementBatch` whose structures carry the plan fingerprint and
  their configuration id, `evaluate()` returns a `ForceBatch` bound to the same fingerprint,
  and a `ForceBatch` may carry its rows in any configuration-id order. `reap()` accepts only
  that object; the reconstructed force constants record the plan fingerprint and schema
  version in their metadata.
- The saturated integer kernel is returned in a canonical column form (Hermite normal form
  of the lattice), so `exact_lattice_basis` is the same integers for equivalent unimodular
  primitive cells and for a re-parameterized kernel basis. The arbitrary-size integer to
  int64 boundary of that layer is documented.
- A `reference` dependency group holds the phonopy and phono3py oracles. The tests that
  compare against them skip cleanly when the group is not installed, and the full suite runs
  in the declared reference environment.

### Removed

- The reference-resolved `cutoff=None` is gone from `InteractionSpace`,
  `FiniteDifferenceCalculation`, `ForceConstantFitter`, `SSCHA` and `resolve_primitive_cutoff`.
  A positive number is a distance in angstrom, a negative integer is a primitive
  neighbour-shell index, and `None` raises with the requirement spelled out. Whether a
  reference identifies the model is now decided only by realization identifiability, which
  raises `InteractionAliasingError` rather than shortening the model. The tutorials state the
  radius their case resolved as an explicit number.
- The experimental scaled orbit-group LASSO is gone, together with its ADMM solver, its
  `solve_scaled_group_lasso` entry point, its convergence logging and the `regularization`,
  `effective_noise_scale`, `active_orbits`, `admm_primal_residual` and `admm_dual_residual`
  result fields. The penalty was defined in column-preconditioned coordinates, which stopped
  meaning the same optimization problem once an orbit's parameters became the coefficients of
  an orthonormal Cartesian basis, and a group penalty is not invariant under a general change
  of parameter coordinates. `ForceConstantFitter.fit()` no longer takes `regularization`:
  passing it is a Python `TypeError` naming the argument, never a silent fallback to least
  squares. A future sparse regularization should be defined on the physical parameters, where
  $\lVert Q\theta\rVert_F = \lVert\theta\rVert_2$ holds.
- The old finite-difference force inputs are gone: a bare `ndarray`, a positional sequence and
  a mapping keyed by numeric configuration id are rejected with a message that says to re-sow,
  because none of them carries a plan fingerprint. A batch whose fingerprint, schema version,
  id set, atom count or shape does not match the plan fails before any differentiation. There
  is no compatibility mode, and an old `sow()` result cannot be reaped at all.
- `research/ase_calculator/prototype.py` is deleted: it imported the removed
  `mlfcs.fitting.backends.wick` package and JAX, none of which is a project dependency, so it
  could not run in the declared environment. Its conclusions and numbers remain in
  `research/ase_calculator/results.json` and the adjoining notes.
- The earlier entry claiming that `cutoff=None` and the scaled group LASSO were untouched by
  the integer-lattice refactor is superseded: this round removes both deliberately.

### Fixed

- `LatticeFrame` documents its three index notions unambiguously: the canonical algebra site,
  the source fractional coordinate of that atom (`source_positions`, canonical order) and the
  source `Atoms` index (`atom_map`). `source_labels` returns the `atom_map` index and derives
  the translation from `source_positions[site]`; looking it up through `atom_map` would apply
  the canonical permutation twice.
- Observation-row selection is documented and tested as *greedy* max-volume selection: it
  requires an orthonormal basis, enumerates component rows in Cartesian component order,
  breaks volume ties towards the smallest row index, returns ascending rows, refuses a
  selection beyond the documented condition limit, and is invariant under a right
  multiplication of the basis by an orthogonal matrix, under a unimodular primitive rebasing
  and under an atom permutation of the source cell.
- Tutorials: the four logs that still carried JAX/CUDA fallback output are regenerated on this
  code, every affected fitting and finite-difference task has a fresh `fit.log` written in
  overwrite mode by its own script, and the Si NEP case stopped reading two `FittingResult`
  fields that no longer exist (the metrics now come from the training dataset). Line-ending-only
  re-writes were dropped instead of committed.


## 4.0.0a5 — 2026-08-24

### Changed

- Renamed the finite-difference workflow to `FiniteDifferenceCalculation`; the previous names are
  removed without compatibility aliases.
- Added one public `perturb_structures()` entry point for Cartesian Gaussian and harmonic-mode
  sampling. SSCHA now uses the same internal harmonic sampler.
- Flattened fitting, rotational, and SSCHA diagnostics into their result records.
- Standardized package reporting on the `mlfcs` logger, with `INFO` and above sent to stdout by
  default and `DEBUG` controlled through standard Python logging.
- Reduced the top-level namespace to the documented public workflow functions and classes.

## 4.0.0a2 — 2026-08-14

### Added

- ALAMODE FCSXML output for combined FC2--FC4, with exact MLFCS atom-order and primitive-cell
  mapping control and documented upstream ALM writer provenance.

### Fixed

- High-order validation prediction now streams the same bounded physical-design groups used by
  Gram construction instead of capturing the full interaction parameterization as multi-gigabyte
  JAX constants during lowering.

## 4.0.0a1 — 2026-08-14

### Added

- Native compact-FC2 commensurate-q sampling for quantum and classical harmonic ensembles.
- Explicit imaginary-mode policies, frequency filtering, sampling diagnostics, and optional
  per-atom radial displacement clipping; clipping is disabled by default.
- Analytic harmonic-model tests and an independent development-only phonopy sampling reference.
- An end-to-end KCl SSCHA reference using phonopy's official pypolymlp potential and fixtures.
- A paired algorithm note that specifies the finite-difference orbit-completeness and
  displacement-key compression contract, including its numerical-stability limits.

### Changed

- `mlfcs.sscha` now fits FC2 with the shared MLFCS symmetry-reduced Gram fitter and writes through
  the shared force-constant I/O layer; phonopy and symfc are no longer runtime dependencies.
- Canonical iterations derive independent reproducible child seeds. Cartesian initialization no
  longer reports a statistically undefined SSCHA free energy.
- Orbit discovery now canonicalizes every cutoff-valid seed before representative deduplication,
  preventing periodic-boundary orbits from being dropped when the anchored candidate set is not
  closed under canonicalization.
- Label-symmetric full-rank pivots reuse all force responses from each displaced structure. The
  corrected K4As4Pt2 maximum-MIC FC3 plan contains 4244 structures instead of the earlier 6636
  redundant plan; the defective intermediate 4160-structure plan is explicitly rejected.

### Compatibility

- Finite-difference `sow()` plans generated before 4.0 must not be combined with 4.0 `reap()`.
  Regenerate the complete ordered structure list when the plan count or order changes.

## 3.1.0 — 2026-08-09

### Added

- Optional harmonic Born-Huang rotational sum rules through
  `rotational_sum_rule=True`, with explicit rejection for higher orders until a joint-order API
  can represent their coupled constraints.
- Joint sparse LSMR projection of translational and rotational constraints.
- Phonopy-style maximum drift reports before and after sum-rule projection.
- `cutoff=None` for the maximum interaction radius enumerable by the current supercell.
- Paired English and Chinese sum-rule documentation.

### Changed

- Translational ASR now uses one sparse, matrix-free LSMR path at every parameter-space size;
  dense Gram construction and its size threshold were removed.

## 3.0.0 — 2026-08-03

Version 3.0 is a complete ASE-first redesign of MLFCS. It replaces the earlier order-specific
implementation with one order-parameterized API and numerical pipeline.

### Added

- Unified finite-difference force-constant calculations for every `order >= 2`.
- Direct ASE Calculator execution and deterministic external `sow()` / `reap()` workflows.
- Recursive central-difference stencils and displacement-key deduplication.
- Sparse symmetry-expanded force constants with lazy dense materialization.
- Strict translational acoustic sum-rule projection using Gram null spaces and sparse LSMR.
- CPU/GPU selection for JAX-accelerated high-rank tensor operations.
- Generic sparse HDF5 output for arbitrary order.
- Dense phonopy FC2, phono3py FC3 HDF5, and ShengBTE FC3/FC4 output.
- ShengBTE FC3/FC4 export with explicit periodic geometry.
- Optional phonopy/symfc stochastic effective-harmonic workflow.
- Independent scientific references against phonopy, phono3py, hiphive-converted data,
  ShengBTE files, and an analytically differentiated FCC Morse FC4 model.
- Serial CI on Python 3.12 and 3.13 with separate unit, package, and scientific-reference jobs.

### Changed

- Public interfaces now use ASE `Atoms` and user-owned ASE calculators.
- Force generation is no longer coupled to a particular electronic-structure or machine-learning
  backend.
- Neighbor selection uses one shared periodic cluster geometry for reconstruction and faithful
  export.
- The command-line interface is removed; version 3.0 is a Python API package.

### Compatibility

- Earlier MLFCS scripts must migrate to `FiniteDifferenceCalculation`, `sow()`, `reap()`, or `run()`.

## Earlier releases

Tags before `v3.0.0` belong to the legacy implementation or development snapshots. They are kept
for provenance but are not covered by the version 3 API contract.
