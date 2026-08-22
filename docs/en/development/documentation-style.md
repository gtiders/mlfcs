---
title: Documentation style
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Documentation style

Define page templates, terminology, status metadata, bilingual mirroring, code verification, links, and dollar-delimited mathematics.

## Sources of truth

- Language home pages own the project introduction and generate the README core.
- Theory owns derivations; Concepts owns the object model; Tutorials owns learning workflows; How-to owns focused tasks.
- Reference owns public signatures; Examples owns reproducible evidence; Roadmap owns unimplemented work.

## Page templates

Theory pages use Motivation, Definitions, Derivation, Implementation in MLFCS, Numerical considerations, Related pages, and References. Tutorials use Goal, Prerequisites, Setup, Steps, Results, Interpretation, Common problems, and Next steps. How-to pages use Problem, Solution, Explanation, Caveats, and Related pages.

## Mathematics and language

Markdown mathematics uses `$...$` inline and `$$...$$` for display equations. English and Chinese pages share paths, heading hierarchy, formulas, code, status, and example meaning; translation must not alter numerical conventions.

## Maintenance

Public API changes update Reference and every affected workflow in the same commit. Algorithm changes update Theory, Validation, and the corresponding example evidence before release.
