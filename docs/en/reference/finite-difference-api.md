---
title: Finite Difference API
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# Versioning Policy

How MLFCS versions its public interface and stored artifacts: semantic versioning of the Python package, stability tiers of top-level exports (stable/experimental/planned), and compatibility promises for the HDF5 archive format.

This page explains what changes may occur within patch, minor and major releases, how deprecations are announced and removed, and which file formats readers must support forever. The `code_verified` field present in the front matter of every documentation page ties each document to the last release where its contents were checked against the code.

## How reconstruction recovers parameters

Each orbit's `observation_rows` are the component rows the plan has to observe, and reconstruction
solves $Q_{\mathrm{obs}}\theta = y_{\mathrm{obs}}$ explicitly, where $Q_{\mathrm{obs}}$ is that
orbit's observation matrix and $y_{\mathrm{obs}}$ are the finite-difference derivative components.
The solution $\theta$ consists of coefficients of $Q$ and is expanded by
`expand_primitive_parameters` into force constants with site and integer-translation labels. These rows
are not parameter pivots: an observed component generally differs from a parameter, and the difference
is controlled by the condition number of $Q_{\mathrm{obs}}$, stored per orbit as
`observation_condition`.
