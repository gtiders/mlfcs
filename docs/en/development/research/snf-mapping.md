---
title: Complete SNF mapping research
audience:
  - developer
status: research
code_verified: 4.0.0a4
---

# Can complete SNF unify finite-supercell mapping?

## Context

Complete Smith normal form can express the translation quotient as finite cyclic coordinates and can clarify direct/reciprocal group structure. In the current $3\times3$ workload it adds transformation bookkeeping without replacing the need for real-space representatives, while the available Python interfaces do not provide a sufficiently stable complete decomposition for the production mapping chain. HNF remains the sole production quotient reduction and representative system. SNF remains a research diagnostic and is not written into IFC keys, HDF5, q-point APIs, or atom mapping.

## Responsibilities

This page defines internal architecture, not additional public API. Structure and interaction layers must not depend upward on fitting, physics workflows, or IO.

## Implementation principles

Represent physical identity explicitly, construct expensive mappings once, and keep format restrictions at writer boundaries.

## Validation

Migrations preserve parameter ordering, force predictions, and IFC semantics, with dependency tests, unit tests, and material regressions.

## Maintenance

Remove superseded paths rather than retaining indefinite compatibility branches. Research prototypes remain separate until numerical and resource acceptance.
