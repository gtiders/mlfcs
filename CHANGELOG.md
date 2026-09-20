# Changelog

[中文](CHANGELOG_ZH.md)

All notable changes are documented here. Releases follow semantic versioning.

## 4.0.0a6 — 2026-09-20

### Removed

- The reference-resolved cutoff (`cutoff=None`) is gone: `InteractionSpace`, `FiniteDifferenceCalculation`
  and `ForceConstantFitter` now require a distance or a negative neighbour-shell index, and the SSCHA
  solver takes a distance too. Aliasing from a badly chosen radius is caught by
  `validate_realization_identifiability`, which raises instead of folding interactions into one
  cluster; the tutorials carry the radius their reference resolves as an explicit number taken from
  their own committed metadata.

- The scaled orbit-group LASSO fitting option is gone: `ForceConstantFitter.fit()` no longer takes
  `regularization`, and `FittingResult` no longer carries `regularization`,
  `effective_noise_scale`, `active_orbits`, `admm_primal_residual` or `admm_dual_residual`.
  `solve_scaled_group_lasso` is deleted. Fitting is the force-only least-squares solve of the
  streamed Gram system; a reference that does not identify its parameters is rejected while it is
  built rather than regularized away.

### Changed

- The exact rank used by identifiability is now proven in both directions by modular arithmetic: a full rank modulo a prime settles it, and a deficient verdict adds primes until their product exceeds the exact Hadamard bound of the largest minors (which guarantees one good prime). The fraction-free elimination that used to settle the deficient verdict is removed, so a rejected reference is reported in milliseconds rather than seconds to minutes. The primes themselves come from `sympy.ntheory.generate.prevprime` instead of a hand-written primality test, which is both faster and one less thing to own.

- Primitive orbit bases are decided in the lattice (scaled) frame. spglib rotations are integer
  matrices there for every cell, so the invariant kernel, its dimension and the orbit basis come
  from exact integer arithmetic instead of a Cartesian eigenvalue threshold.
  `PrimitiveInteractionOrbit.basis` is now that integer basis and the fitted parameters are its
  coefficients, with no canonicalization; consumers render physical tensors through the cell once,
  so reconstructed force constants are unchanged (verified to a relative $2\times10^{-16}$) while
  parameter coordinates differ. The orbit entry points (`build_primitive_interaction_space`,
  `generated_orbit`, `traverse_indexed_orbit`) no longer take a `tolerance`, and
  `RealizedInteractionSpace` carries the `cell` that its orbits are expressed in.
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

- The acoustic-sum-rule coefficient guard and de-duplication are unchanged after
  measurement: the rows are Cartesian components of algebraic numbers, so "the same
  constraint" cannot be decided by an integer key, and removing the guard lets one
  constraint's rows diverge in support, which moved a constrained fit by a factor of two.
  The guard separates algebraic zeros from genuine coefficients with orders of magnitude to
  spare, and the reason is documented next to it.
- The `cutoff=None` shell resolver finds an exactly attained periodic boundary through
  `np.nextafter(shortest length, inf)` instead of a `+1e-8 Å` padding, and its margin below
  the aliasing boundary is unchanged.
- Identifiability gives the same answer for every cell. `validate_realization_identifiability`
  ranks the integer lattice-frame realization matrix and decides a deficient component with exact
  fraction-free elimination, so a coefficient can no longer be dropped by a filter or rounded into
  an integer matrix. Primitive cells whose Cartesian rotations are irrational (an fcc primitive
  $60^\circ$ cell, hexagonal and rhombohedral cells) build their orbits and realize their
  interactions instead of failing.
- `PeriodicGeometry` now resolves the minimum image through ASE's Minkowski-reduction based search
  instead of `ase.geometry.find_mic`. `find_mic` skips the reduction whenever the folded vector is
  shorter than `0.5 * min(cell.lengths())`, and that bound is not the inradius of the Wigner-Seitz
  cell, so skewed cells received a non-minimum image. Minimum-image lengths, degenerate image sets,
  and cluster image selection now agree with the true minimum image for skewed and unimodularly
  transformed frames; the `mic()` calling convention is unchanged.

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
