# 有限差分

`ForceConstantCalculation` 接收显式 primitive 和参考超胞。需要从矩阵准备结构时，先调用
`mlfcs.tools.build_supercell`。`sow()` 按参考顺序返回结构；`reap()` 要求力保持该顺序，也可以使用
configuration ID 映射。

```python
calculation = ForceConstantCalculation(
    primitive, reference=reference_supercell, order=3, cutoff=-4, displacement=0.01
)
structures = calculation.sow()
forces = [calculator.get_forces(atoms) for atoms in structures]
result = calculation.reap(forces, acoustic_sum_rule=True)
```

计算前会去重等价的中心差分位移。有限差分不会静默重排参考超胞；外部程序请使用带 manifest 的工作流。
