---
title: Finite differences
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Finite differences

`FiniteDifferenceCalculation` accepts an explicit primitive and reference supercell. Use
`mlfcs.build_supercell` before constructing the calculation when a matrix-based structure
must be prepared. `sow()` returns structures in reference order; `reap()` requires forces in
that same order (or a configuration-ID mapping).

```python
calculation = FiniteDifferenceCalculation(
    primitive,
    reference=reference_supercell,
    order=3,
    cutoff=-4,
    displacement=0.01,
)
structures = calculation.sow()
forces = [calculator.get_forces(atoms) for atoms in structures]
result = calculation.reap(forces, acoustic_sum_rule=True)
```

Central-difference keys are deduplicated before evaluation. Finite differences do not silently
reorder the reference supercell. For an external calculation, use the manifest-based workflow.
