# 有限差分

`ForceConstantCalculation` 接收显式 primitive，并要求提供超胞矩阵或参考超胞二者之一。`sow()` 按参考
顺序返回结构；`reap()` 要求力保持该顺序，也可以使用 configuration ID 映射。

```python
calculation = ForceConstantCalculation(
    primitive, order=3, supercell_matrix=(2, 2, 3), cutoff=-4, displacement=0.01
)
structures = calculation.sow()
forces = [calculator.get_forces(atoms) for atoms in structures]
result = calculation.reap(forces, acoustic_sum_rule=True)
```

计算前会去重等价的中心差分位移。有限差分不会静默重排参考超胞；外部程序请使用带 manifest 的工作流。
