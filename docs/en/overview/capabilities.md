---
title: Capabilities and boundaries
audience:
  - beginner
status: stable
code_verified: 4.0.0a4
---

# Capabilities and boundaries

This page records stable, experimental, planned, and unsupported capabilities against the current public API. Feature status is a contract and is updated with code changes.

## Stable

- Explicit primitive/reference structure relations, HNF periodic indexing, arbitrary atom order, and non-diagonal integer supercells.
- Symmetry-reduced finite differences and force-only fitting from FC2 through supported higher orders.
- Taylor output, Wick fitting coordinates, translational constraints, and FC2 Born–Huang/Huang correction.
- Sparse native HDF5 v3, target-supercell realization, and the documented external writers.

## Experimental

- FC4 loop SCPH.
- Stochastic effective-harmonic SSCHA.

These workflows return complete effective FC2 objects but require explicit convergence and physical validation.

## Planned

- Long-range dipole-force subtraction before short-range fitting.
- Signed-displacement symmetry reduction for finite differences.

## Not supported

- Joint fitting of data from different supercells.
- Reading legacy native HDF5 schemas.
- Rigid Cartesian structure rotation during export.
- Explicit FC3 bubble self-energy or multipole electrostatic correction.
