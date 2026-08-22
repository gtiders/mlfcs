---
title: Documentation style
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Documentation style

## Context

Define page templates, terminology, status metadata, bilingual mirroring, code verification, links, and dollar-delimited mathematics. - Language home pages own the project introduction and generate the README core. - Theory owns derivations; Concepts owns the object model; Tutorials owns learning workflows; How-to owns focused tasks. - Reference owns public signatures; Examples owns reproducible evidence; Roadmap owns unimplemented work. Theory pages use Motivation, Definitions, Derivation, Implementation in MLFCS, Numerical considerations, Related pages, and References. Tutorials use Goal, Prerequisites, Setup, Steps, Results, Interpretation, Common problems, and Next steps. How-to pages use Problem, Solution, Explanation, Caveats, and Related pages. Markdown mathematics uses `$...$` inline and `$$...$$` for display equations. English and Chinese pages share paths, heading hierarchy, formulas, code, status, and example meaning; translation must not alter numerical conventions. Public API changes update Reference and every affected workflow in the same commit. Algorithm changes update Theory, Validation, and the corresponding example evidence before release.

## Responsibilities

This page defines internal architecture, not additional public API. Structure and interaction layers must not depend upward on fitting, physics workflows, or IO.

## Implementation principles

Represent physical identity explicitly, construct expensive mappings once, and keep format restrictions at writer boundaries.

## Validation

Migrations preserve parameter ordering, force predictions, and IFC semantics, with dependency tests, unit tests, and material regressions.

## Maintenance

Remove superseded paths rather than retaining indefinite compatibility branches. Research prototypes remain separate until numerical and resource acceptance.
