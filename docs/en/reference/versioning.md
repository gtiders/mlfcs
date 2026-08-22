---
title: Versioning policy
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Versioning policy

## Purpose

Define public API stability, alpha status, HDF5 schema compatibility, and documentation verification versions.

## Stability

Reference pages describe current public behavior. Defaults, units, returns, and exceptions match the verified version; planned features are excluded.

## Diagnosis

Read the exception and diagnostics first, then verify structure and array shapes. Use Theory for physical approximations and How-to for workflows.

## Compatibility

Internal module paths are not public contracts. Import public objects from `mlfcs` and record the package version and schema with saved results.
