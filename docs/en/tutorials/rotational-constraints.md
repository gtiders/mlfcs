---
title: Rotational constraints
audience:
  - advanced
status: stable
code_verified: 4.0.0a4
---

# Rotational constraints

## Goal

Compare raw, ASR, Born–Huang, and Huang-corrected FC2 on two-dimensional material examples.

## Prerequisites

Prepare a validated primitive, reference supercell, and force source. All snapshots retain the reference atom labels and ordering.

## Workflow

Run input checks, calculation or fitting, diagnostics, IFC storage, and downstream validation as separate steps so expensive force evaluations are reusable.

## Expected result

Success includes sensible parameter or displacement counts, acceptable residuals, and interpretable physical results in the target representation—not merely an output file.

## Common problems

Exclude atom-order, unit, cutoff, finite-supercell, and force-convergence problems before changing constraints or numerical settings.
