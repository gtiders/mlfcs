---
title: Choose a cutoff
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Choose a cutoff

## Problem

Choose a positive distance, a negative neighbor shell, or the documented `None` radius while recognizing long-range and finite-reference limits.

## Procedure

Fix the primitive, reference, and target output; record orders, cutoffs, body order, constraints, and units. Verify every input structure against the reference before running.

## Checks

Inspect structure matching, orbit and parameter counts, rank, residuals, and warnings. Validate IFCs on an explicit target supercell instead of merely checking that a file exists.

## Rationale

MLFCS separates physical IFC identity from file ordering. Explicit structures prevent silent reordering, periodic-image ambiguity, and downstream-default mismatches.

## Limitations

This guide does not replace convergence tests for electronic-structure forces or downstream solvers. Long-range electrostatics and insufficient training coverage require physical remedies.
