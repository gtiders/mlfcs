# Changelog

[中文](CHANGELOG_ZH.md)

All notable changes are documented here. Releases follow semantic versioning.

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
- Faithful ShengBTE periodic geometry by default and an explicit thirdorder compatibility mode.
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

- Existing thirdorder sow order and ShengBTE layout are available only when explicitly requested.
- Earlier MLFCS scripts must migrate to `ForceConstantCalculation`, `sow()`, `reap()`, or `run()`.

## Earlier releases

Tags before `v3.0.0` belong to the legacy implementation or development snapshots. They are kept
for provenance but are not covered by the version 3 API contract.
