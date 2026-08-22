---
title: Single-reference fitting
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Single-reference fitting

## Context

Record the deliberate restriction to one fixed reference supercell per fit while allowing post-fit realization on other targets.

## Responsibilities

This page defines internal architecture, not additional public API. Structure and interaction layers must not depend upward on fitting, physics workflows, or IO.

## Implementation principles

Represent physical identity explicitly, construct expensive mappings once, and keep format restrictions at writer boundaries.

## Validation

Migrations preserve parameter ordering, force predictions, and IFC semantics, with dependency tests, unit tests, and material regressions.

## Maintenance

Remove superseded paths rather than retaining indefinite compatibility branches. Research prototypes remain separate until numerical and resource acceptance.
