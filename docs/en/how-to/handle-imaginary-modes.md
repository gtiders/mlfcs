---
title: Diagnose imaginary modes
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Diagnose imaginary modes

## Problem

Separate structural instability from cutoff, long-range electrostatics, mapping, fitting, and constraint problems before changing a model.

## Procedure

Fix the primitive, reference, and target output; record orders, cutoffs, body order, constraints, and units. Verify every input structure against the reference before running.

## Checks

Inspect structure matching, orbit and parameter counts, rank, residuals, and warnings. Validate IFCs on an explicit target supercell instead of merely checking that a file exists.

## Rationale

MLFCS separates physical IFC identity from file ordering. Explicit structures prevent silent reordering, periodic-image ambiguity, and downstream-default mismatches.

## Limitations

This guide does not replace convergence tests for electronic-structure forces or downstream solvers. Long-range electrostatics and insufficient training coverage require physical remedies.
