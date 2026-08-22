---
title: Native HDF5 v3
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Native HDF5 v3

## Context

Record the exact-$R$ sparse schema and the decision to reject older native schemas instead of guessing their atom semantics.

## Responsibilities

This page defines internal architecture, not additional public API. Structure and interaction layers must not depend upward on fitting, physics workflows, or IO.

## Implementation principles

Represent physical identity explicitly, construct expensive mappings once, and keep format restrictions at writer boundaries.

## Validation

Migrations preserve parameter ordering, force predictions, and IFC semantics, with dependency tests, unit tests, and material regressions.

## Maintenance

Remove superseded paths rather than retaining indefinite compatibility branches. Research prototypes remain separate until numerical and resource acceptance.
