# Finite differences

`ForceConstantCalculation` accepts an explicit primitive and either a supercell matrix or a
reference supercell. `sow()` returns structures in reference order; `reap()` requires forces in
that same order (or a configuration-ID mapping).

```python
calculation = ForceConstantCalculation(
    primitive,
    order=3,
    supercell_matrix=(2, 2, 3),
    cutoff=-4,
    displacement=0.01,
)
structures = calculation.sow()
forces = [calculator.get_forces(atoms) for atoms in structures]
result = calculation.reap(forces, acoustic_sum_rule=True)
```

Central-difference keys are deduplicated before evaluation. Finite differences do not silently
reorder the reference supercell. For an external calculation, use the manifest-based workflow.
