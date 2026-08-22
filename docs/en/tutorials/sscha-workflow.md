---
title: SSCHA workflow
audience:
  - advanced
status: experimental
code_verified: 4.0.0a4
---

# SSCHA workflow

## Goal

Generate harmonic ensembles with an ASE calculator, apply linear mixing, and inspect effective FC2 and free-energy diagnostics.

## Prerequisites

Prepare a validated primitive, reference supercell, and force source. All snapshots retain the reference atom labels and ordering.

## Workflow

Run input checks, calculation or fitting, diagnostics, IFC storage, and downstream validation as separate steps so expensive force evaluations are reusable.

## Expected result

Success includes sensible parameter or displacement counts, acceptable residuals, and interpretable physical results in the target representation—not merely an output file.

## Common problems

Exclude atom-order, unit, cutoff, finite-supercell, and force-convergence problems before changing constraints or numerical settings.
