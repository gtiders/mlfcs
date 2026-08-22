---
title: Apply sum rules
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Apply sum rules

## Problem

Apply translational constraints during reconstruction or fitting and optional FC2 Born–Huang/Huang post-processing through the public function.

## Procedure

Fix the primitive, reference, and target output; record orders, cutoffs, body order, constraints, and units. Verify every input structure against the reference before running.

## Checks

Inspect structure matching, orbit and parameter counts, rank, residuals, and warnings. Validate IFCs on an explicit target supercell instead of merely checking that a file exists.

## Rationale

MLFCS separates physical IFC identity from file ordering. Explicit structures prevent silent reordering, periodic-image ambiguity, and downstream-default mismatches.

## Limitations

This guide does not replace convergence tests for electronic-structure forces or downstream solvers. Long-range electrostatics and insufficient training coverage require physical remedies.
