---
title: Graphene rotational constraints
audience:
  - user
status: stable
code_verified: 4.0.0a4
examples:
  - examples/fitting/graphene
---

# Graphene rotational constraints

## Objective

A second two-dimensional validation of ASR, Born–Huang, and Huang behavior.

## Inputs and provenance

The case directory retains the structures, settings, scripts, and permitted reference data needed for reproduction. Third-party data must identify its source.

## Execution

Read the case README, then run preparation, calculation/fitting, analysis, and plotting scripts separately. Cases do not depend on scripts in other cases.

## Checks

Record orbit/parameter or displacement counts, errors, constraint residuals, IFC files, and figures. Inspect bands for NaNs, scale errors, discontinuities, and unexpected imaginary modes.

## Regression role

The case validates physics, not only file creation. Internal ordering may change; IFCs, predicted forces, and derived observables in the same representation should not.

## Case directory

[examples/fitting/graphene](https://github.com/gtiders/mlfcs/tree/dev/examples/fitting/graphene) contains the authoritative inputs, independent scripts, retained reference data, and generated-result policy for this case.
