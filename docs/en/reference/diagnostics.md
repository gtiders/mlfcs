---
title: Diagnostics
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Diagnostics

## Purpose

When a result is unexpected, check the structure relation first: primitive atom count, reference atom order, supercell matrix, and maximum mapping residual. Then check cutoff shell, force units, ASR residual, and the target writer's required supercell. For SCPH, inspect `result.history`, the final RMS frequency change, and negative squared frequencies. A non-converged effective FC2 is diagnostic output, not a production result.

## Stability

Reference pages describe current public behavior. Defaults, units, returns, and exceptions match the verified version; planned features are excluded.

## Diagnosis

Read the exception and diagnostics first, then verify structure and array shapes. Use Theory for physical approximations and How-to for workflows.

## Compatibility

Internal module paths are not public contracts. Import public objects from `mlfcs` and record the package version and schema with saved results.
