# 有限差分 API

`FiniteDifference` 生成确定且按阶次定义的 ASE 结构序列。该序列同时是重建的数据契约：`displacements()` 返回的第 i 个结构，必须与该几何和原子顺序对应的力结果配对。

## 定义原胞模型和参考超胞

```python
import numpy as np
from ase.build import bulk
from mlfcs import (
    PrimitiveCell, Supercell, ClusterMap, build_cluster_space, FiniteDifference,
)

primitive_atoms = bulk("Al", "fcc", a=4.05)
primitive = PrimitiveCell.from_atoms(primitive_atoms, symprec=1e-5)
space = build_cluster_space(
    primitive,
    cutoffs={2: 4.0},
    max_body_orders={2: 2},
)
supercell_atoms = primitive_atoms.repeat((3, 3, 3))
supercell = Supercell.from_atoms(
    primitive, supercell_atoms, matrix=np.diag([3, 3, 3])
)
mapping = ClusterMap.build(space, supercell)
mapping.rank_info(2).require_full()
```

`symprec` 是以 Å 为单位声明的几何容差，用于对称性识别和原胞到超胞的映射。整数超胞矩阵必须显式提供。调用方提供参考超胞，并负责判断其大小是否足以辨识所需相互作用。

## 生成位移并计算

```python
from ase.calculators.emt import EMT

fd = FiniteDifference(mapping, order=2, disps=(0.01, 0.02))
displaced = fd.displacements()
evaluated = fd.evaluate(EMT())
fc2 = fd.reconstruct(evaluated)
```

这里用 ASE 的 EMT 展示 calculator 调用。只要 calculator 支持结构中的元素和条件，也可以替换成任意 ASE calculator。`evaluate(calculator)` 会明确对每个结构重新计算力，并将结果保存在返回结构的 single-point calculator 中。

构造函数为：

```python
FiniteDifference(mapping, *, order: int, disps: float | Sequence[float] = 0.01)
```

- `order` 是力常数阶次，必须至少为 2。
- `disps` 是一个以 Å 为单位的正位移长度，或一组互不相同的正位移长度。使用多个位移时，按位移平方的偶次误差多项式外推到零位移。
- 对应阶次必须存在于 cluster space 中，且超胞映射必须在结构上可辨识。

## 外部计算器和保存的结构

也可以由外部程序计算，而不调用 `fd.evaluate`。将 `fd.displacements()` 输出为能保留原子顺序与晶胞的格式，完成外部计算后，按相同顺序读回结构，并将每个力数组附到对应 ASE 对象：

```python
from ase.calculators.singlepoint import SinglePointCalculator

for atoms, forces in zip(displaced, force_arrays, strict=True):
    atoms.calc = SinglePointCalculator(atoms, forces=forces)

fc2 = fd.reconstruct(displaced)
```

此处的 `force_arrays` 表示外部计算得到的力，每个结构对应一个形状为 `(n_atoms, 3)` 的数组。`reconstruct` 不直接接收这些数组：ASE 结构还承载必要的几何、原子顺序和力的关联信息。可使用 extxyz 等有序格式保存帧；不要排序、去重或以其他方式重排。

## 重建契约与常见错误

`fd.reconstruct(structures)` 只接受带有已存储力的有序 ASE `Atoms` 序列。它会检查序列长度、元素及原子顺序、周期性、晶胞和周期位置，并验证力数组是否有限。遇到不匹配时会拒绝，不会猜测原子置换。

常见问题：

- `AliasingError`：显式参考超胞无法区分所有目标原胞参数。应更换参考超胞，或降低模型截断/体阶。
- 缺少已存储力：将力结果附到每个 ASE 结构；重建不会调用 calculator。
- 几何或顺序不匹配：恢复原帧序和原子顺序后再重建。

结果是只包含所选阶次的原胞 `ForceConstants` 对象。不同阶次可通过 `ForceConstants.combine` 合并；也可用 `fc2.save("fc2.mlfcs")` 保存原生格式，或通过 `fc2.write(...)` 导出。
