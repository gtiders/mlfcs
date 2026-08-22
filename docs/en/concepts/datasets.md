---
title: Data objects
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Data objects

## Definition

A displacement plan, a fitting dataset, and an SSCHA ensemble are different domain objects and do not share one generic dataset abstraction.

## Why it matters

“Data objects” determines the physical identity carried between calculation stages. Mixing structure labels, periodic translations, numerical arrays, or workflow objects causes mismatches and double counting.

## Relations

Structure mapping defines atoms and translations; interactions and orbits build on it; parameterization connects independent variables to physical IFCs. Workflows must consume these objects rather than infer mappings.

## When users need it

Check this concept while preparing a reference, choosing cutoffs, interpreting parameter counts, exporting to a target supercell, or diagnosing differences between programs.

## Validation rule

Compare structure labels, integer translations, tensors, and final physical arrays—not file order, internal indices, or cache order.
