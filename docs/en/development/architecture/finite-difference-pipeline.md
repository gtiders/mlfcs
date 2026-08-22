---
title: Finite-difference pipeline
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Finite-difference pipeline

## Context

Trace deterministic signed configurations through force collection, recursive reconstruction, constraints, and sparse expansion.

## Responsibilities

This page defines internal architecture, not additional public API. Structure and interaction layers must not depend upward on fitting, physics workflows, or IO.

## Implementation principles

Represent physical identity explicitly, construct expensive mappings once, and keep format restrictions at writer boundaries.

## Validation

Migrations preserve parameter ordering, force predictions, and IFC semantics, with dependency tests, unit tests, and material regressions.

## Maintenance

Remove superseded paths rather than retaining indefinite compatibility branches. Research prototypes remain separate until numerical and resource acceptance.
