---
title: Parameterization
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Parameterization

## Context

Describe invariant tensor bases, physical columns, Wick coordinates, constraint reduction, and Taylor expansion.

## Responsibilities

This page defines internal architecture, not additional public API. Structure and interaction layers must not depend upward on fitting, physics workflows, or IO.

## Implementation principles

Represent physical identity explicitly, construct expensive mappings once, and keep format restrictions at writer boundaries.

## Validation

Migrations preserve parameter ordering, force predictions, and IFC semantics, with dependency tests, unit tests, and material regressions.

## Maintenance

Remove superseded paths rather than retaining indefinite compatibility branches. Research prototypes remain separate until numerical and resource acceptance.
