# MLFCS

MLFCS 从 ASE 结构和原子力构造原胞力常数。公共工作流由三个明确对象组成：

1. `PrimitiveCell` 与 `build_cluster_space` 定义原胞基元、对称性、相互作用截断和参数空间。
2. `Supercell` 与 `ClusterMap` 将原胞模型映射到一个明确给定的参考超胞。
3. `FiniteDifference` 重建单个阶次；`FitSystem` 构建并求解多个阶次的仅力拟合。

结构长度使用 Å，能量使用 eV，力使用 eV/Å。n 阶力常数的单位是 eV/Åⁿ。

## 选择工作流

- [有限差分](finite-difference-api.md)：生成有序位移结构，调用 ASE calculator，并重建一个阶次。
- [力拟合](fitting-api.md)：将带有已存储原子力的 ASE 结构流式写入可复用拟合系统，并联合求解多个阶次。
- [Q&A](Q&A.md)：了解力的存储、顺序、可辨识性、求解器和常见选择。

## 最小模型设置

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
```

参考超胞是明确的数据输入；MLFCS 不会静默选择或扩展它。在生成昂贵的力之前，应通过 `mapping.rank_info()` 检查它能否辨识所需模型。有限差分和拟合路径分别见对应 API 文档。

## 范围

每个 `FiniteDifference` 对象处理一个阶次；一个 `FitSystem` 可以处理同一 `ClusterSpace` 中的所有阶次。力常数属于原胞模型；超胞 realization 和外部文件格式都是派生操作。
