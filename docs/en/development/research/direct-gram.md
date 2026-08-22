---
title: Direct-Gram research
audience:
  - developer
status: research
code_verified: 4.0.0a4
---

# Direct-Gram and matrix-free design research

## Context

This research tested exact accumulation of $G=X^TX$ and $b=X^Ty$ without materializing the current streamed design batches. The evaluated interaction-streaming, feature-factorization, and finite-sample moment approaches did not reduce the dominant contraction cost without introducing parameter-squared work or larger intermediates, so the production fitting path remains unchanged. The current result is **No-Go**. The prototype is retained as a mathematical and performance boundary, not as a public API or alternative runtime path.

## Responsibilities

This page defines internal architecture, not additional public API. Structure and interaction layers must not depend upward on fitting, physics workflows, or IO.

## Implementation principles

Represent physical identity explicitly, construct expensive mappings once, and keep format restrictions at writer boundaries.

## Validation

Migrations preserve parameter ordering, force predictions, and IFC semantics, with dependency tests, unit tests, and material regressions.

## Maintenance

Remove superseded paths rather than retaining indefinite compatibility branches. Research prototypes remain separate until numerical and resource acceptance.
