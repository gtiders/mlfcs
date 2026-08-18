# Workflows

All workflows share the same structure relation, sparse IFC model, constraints, and writers.
Choose the source of forces first; the downstream export is a separate step.

| Source | Workflow |
|---|---|
| ASE calculator | [Finite differences](finite-difference.md) |
| VASP/QE/other external code | [External calculators](external-calculators.md) |
| Existing snapshots and forces | [Force-only fitting](fitting.md) |
| Harmonic sampling | [SSCHA](sscha.md) |
| FC2 + FC4 loop correction | [SCPH](scph.md) |
