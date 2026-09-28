# 力拟合

力拟合从带有已存储原子力的 ASE 结构中估计线性力常数参数。`FitSystem` 表示针对一个 `ClusterSpace` 的可复用最小二乘问题；同一空间中配置的所有力常数阶次共同求解。

## `FitSystem`：流式拟合问题

```python
FitSystem.from_atoms(mapping, structures)
system.solve(*, rtol=1e-8, max_steps=1000)
system.force_constants(parameters)
```

| 参数 | 含义 |
| --- | --- |
| `mapping` | 定义原胞模型在训练超胞中实现方式的 `ClusterMap`。 |
| `structures` | 带有已存储力的 ASE `Atoms` 可迭代对象。每帧必须使用 mapping 对应的原子顺序、晶胞和周期几何。 |
| `rtol` | 传给默认列缩放 MINRES 求解的相对收敛精度；默认 `1e-8`。 |
| `max_steps` | MINRES 最大迭代步数；默认 `1000`。 |
| `parameters` | 按 `ClusterSpace` 阶次/块布局排列的完整参数向量。 |

构造时会流式地将结构累积到紧凑正规系统，不保留输入帧。它只读取 ASE 结构上已经存储的力；计算器求力属于数据准备流程。

更底层的值构造函数为 `FitSystem(space, matrix, rhs, force_norm, n_equations, n_structures)`，适用于恢复或组装已计算的正规系统。`space` 定义参数布局；`matrix` 和 `rhs` 分别是 $A^T A$ 与 $A^T f$；`force_norm` 是 $f^T f$；最后两个整数记录方程数和结构数。大多数工作流应使用 `FitSystem.from_atoms(mapping, structures)`，由带力结构累积这些量。

## Si 拟合最小示例

```python
from ase.io import iread, read
from mlfcs import ClusterMap, ClusterSpace, FitSystem, Supercell

space = ClusterSpace(
    read("POSCAR"),
    cutoffs={2: -7, 3: -6, 4: -3, 5: -2},
    max_body_orders={2: 2, 3: 3, 4: 4, 5: 3},
)
supercell = Supercell.from_atoms(space.primitive, read("SPOSCAR"))
mapping = ClusterMap.build(space, supercell)
mapping.rank_info().require_full()

system = FitSystem.from_atoms(mapping, iread("train.xyz", index=":"))
parameters = system.solve(rtol=1e-8, max_steps=10_000)
model = system.force_constants(parameters)
model.save("fc-si-fit.mlfcs")
```

此例对应仓库中的 [Si 拟合脚本](../../tutorial/SI/fitting/fit.py)；该脚本还会施加 ASR 并导出 FC2、FC3。训练力可以来自 DFT 或机器学习势；只要已存储在 ASE `Atoms` 上，拟合接口相同。

## 线性系统与求解器参数

给定设计矩阵 $A$、力向量 $f$ 和参数 $p$，拟合目标是 $min_p \|Ap-f\|_2$。默认路径累积 $H=A^T A$ 和 $g=A^T f$，然后使用列缩放 MINRES 求解正规系统。第 $i$ 列的缩放系数为 $1/\sqrt{H_{ii}}$。

拟合系统公开 `matrix`、`rhs`、`force_norm`、`n_equations`、`n_structures`、`n_parameters`、`unobserved_parameters` 和 `fingerprint`。`residual(parameters)`、`rmse(parameters)` 与 `relative_error(parameters)` 用于评估系统所代表的力残差。`parameter_name(index)` 将参数向量索引转换成阶次、orbit 和分量信息。

若某个对角元 $H_{ii}$ 恰好为零，该参数没有出现在训练设计中。`solve()` 会抛出 `UnobservedParameterError`；应增加有信息量的结构或调整模型。`rtol` 控制迭代收敛，`max_steps` 限制迭代步数。

## 复用和合并拟合系统

`FitSystem` 保存正规矩阵、右端、力范数和计数。兼容的系统可用 `system_a + system_b` 合并；它们的簇空间身份必须一致。拟合系统可以 pickle 以便可信环境中的本地复用。pickle 加载可能执行代码，因此只应加载可信来源的文件。

## 使用 `FitData` 保留原始方程

需要原始设计行、例如希望调用其他最小二乘求解器时，可以选择 `FitData`：

```python
import numpy as np
from mlfcs import FitData

data = FitData.from_atoms(mapping, iread("train.xyz", index=":"))
design, forces = data.arrays()
parameters = np.linalg.lstsq(design, forces, rcond=None)[0]
model = data.force_constants(parameters)
```

直接值构造函数为 `FitData(space, designs, forces)`：`designs` 与 `forces` 分别包含每个结构对应的矩阵/向量。公开方法包括 `from_atoms(mapping, structures)`、`arrays()`、`normal_system()` 和 `force_constants(parameters)`。`designs` 与 `forces` 可读取逐结构数组；`n_structures`、`n_equations` 和 `n_parameters` 报告数据维度。`FitData` 会保留完整设计矩阵，因此内存需求随力方程数量增长；`FitSystem.from_atoms` 则是紧凑的流式路径。

拟合模型的投影与导出见[力常数](force-constants.md)。
