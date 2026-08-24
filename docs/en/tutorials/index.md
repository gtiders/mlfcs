---
title: Workflows
audience:
  - user
status: stable
code_verified: 4.0.0a5
---

# Workflows

All workflows share the same structure relation, sparse IFC model, constraints, and writers.
Choose the source of forces first; the downstream export is a separate step.

| Source | Workflow |
|---|---|
| ASE calculator | [Finite differences](finite-difference-workflow.md) |
| VASP/QE/other external code | [External calculators](external-calculator.md) |
| Existing snapshots and forces | [Force-only fitting](first-fc2-fitting.md) |
| Harmonic sampling | [SSCHA](sscha-workflow.md) |
| FC2 + FC4 loop correction | [SCPH](scph-workflow.md) |
