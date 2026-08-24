---
title: Long-range electrostatics roadmap
audience:
  - advanced
status: planned
code_verified: 4.0.0a5
---

# Long-range electrostatics roadmap

## Current status

Plan force subtraction for dipole electrostatics before short-range fitting and record the additional Born-charge and dielectric inputs.

## Scope

Roadmap material is not a stable API. Equations, prototypes, and research conclusions must not be treated as implemented features.

## Entry criteria

Implementation requires a physical definition, data requirements, canonical-IFC interface, failure semantics, and a reproducible baseline. Experimental compatibility branches are not permanent.

## Acceptance

Require analytic tests, structure/symmetry regressions, material cases, and resource measurements. Classify changed results before updating baselines.

## Non-goals

Do not expand features by silently averaging, projecting, or regularizing unidentifiable information. No-Go findings remain design evidence.
