# Changelog

[中文](CHANGELOG_ZH.md)

All notable changes are documented here. Releases follow semantic versioning.

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
