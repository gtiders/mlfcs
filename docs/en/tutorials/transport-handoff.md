---
title: Transport handoff
audience:
  - user
status: stable
code_verified: 4.0.0a6
---

# Transport handoff

## Goal

Export validated force constants for phono3py or ShengBTE while keeping transport solving outside the MLFCS core.

## Prerequisites

Prepare a validated primitive, reference supercell, and force source. All snapshots retain the reference atom labels and ordering.

## Workflow

Run input checks, calculation or fitting, diagnostics, IFC storage, and downstream validation as separate steps so expensive force evaluations are reusable.

## Expected result

Success includes sensible parameter or displacement counts, acceptable residuals, and interpretable physical results in the target representation—not merely an output file.

## Common problems

Exclude atom-order, unit, cutoff, finite-supercell, and force-convergence problems before changing constraints or numerical settings.
