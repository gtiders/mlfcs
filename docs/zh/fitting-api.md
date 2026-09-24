# 力拟合 API

力拟合是一个仅使用原子力的线性问题。`FitSystem` 消费已经带有力的 ASE 结构，构建优化后的充分正规系统，并联合求解配置的所有阶次。它不会调用 calculator，也不会保留训练结构。

## 构建共享模型和超胞映射

```python
import numpy as np
from ase.build import bulk
from mlfcs import PrimitiveCell, Supercell, ClusterMap, build_cluster_space

primitive_atoms = bulk("Al", "fcc", a=4.05)
primitive = PrimitiveCell.from_atoms(primitive_atoms, symprec=1e-5)
space = build_cluster_space(
    primitive,
    cutoffs={2: 4.0, 3: 3.0},
    max_body_orders={2: 2, 3: 3},
)
supercell_atoms = primitive_atoms.repeat((3, 3, 3))
supercell = Supercell.from_atoms(
    primitive, supercell_atoms, matrix=np.diag([3, 3, 3])
)
mapping = ClusterMap.build(space, supercell)
mapping.rank_info().require_full()
```

每个阶次分别指定截断半径和最大体阶。显式参考超胞属于训练数据几何，不属于原胞 `ClusterSpace`。

## 构建并求解拟合系统

```python
from ase.io import read
from mlfcs import FitSystem

training = read("training.extxyz", index=":")
system = FitSystem.from_atoms(mapping, training)
parameters = system.solve()
force_constants = system.force_constants(parameters)

print(system.n_structures, system.n_parameters)
print("force RMSE:", system.rmse(parameters), "eV/Å")
force_constants.save("fit.mlfcs")
```

每个训练帧必须具有相同的原子顺序、周期参考晶胞，并且 ASE calculator 中已经保存力结果。例如，带力写出的 extxyz 可由 ASE 读取后直接传入。`FitSystem.from_atoms` 只读取已保存结果，不会触发新计算；不接受裸 NumPy 力数组或没有已存储力属性的结构。

精简接口为：

```python
FitSystem.from_atoms(mapping, structures)
system.solve(*, rtol=1e-8, max_steps=1000)
system.force_constants(parameters)
```

默认求解器是在正规系统上使用列缩放 MINRES。参数 i 的精确缩放为 `1 / sqrt(H[i, i])`，其中 `H` 是正规矩阵。若对角元为零，表示训练设计从未观测该参数；求解会抛出具名的 `UnobservedParameterError`，而不会静默正则化。应增加结构信息，或调整模型。

## 复用、合并和检查

`FitSystem` 存储对称矩阵 `H = A.T @ A`、右端项 `g = A.T @ f`、力范数和计数。这些量足以求解和评估最小二乘问题，无需保留庞大的设计矩阵或源结构。

- 用 `combined = system_a + system_b` 相加兼容系统；两者的 cluster-space 指纹必须相同。
- 可 pickle 系统以便可信环境中的本地复用。只反序列化可信来源的 pickle 文件。
- 可检查 `system.matrix`、`system.rhs`、`system.force_norm`、`system.n_equations`、`system.n_structures` 和 `system.unobserved_parameters`。
- 可用 `system.residual(parameters)`、`system.rmse(parameters)` 和 `system.relative_error(parameters)` 评估解。

RMSE 单位为 eV/Å。n 阶力常数单位为 eV/Åⁿ。平移或旋转后处理与拟合分开；例如，`force_constants.enforce_asr()` 会返回投影后的力常数模型及报告。
