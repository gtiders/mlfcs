# MLFCS

[![文档构建](https://github.com/gtiders/mlfcs/actions/workflows/ci.yml/badge.svg)](https://github.com/gtiders/mlfcs/actions/workflows/ci.yml)
[![文档站](https://img.shields.io/badge/docs-GitHub%20Pages-0f766e)](https://gtiders.github.io/mlfcs/zh/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776ab)](https://www.python.org/)
[![许可证：GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

[English](README.md) | 简体中文

<!-- BEGIN GENERATED: docs/zh/index.md -->

MLFCS 从 ASE 结构和原子力构造原胞力常数。公共工作流由三个明确对象组成：

1. `PrimitiveCell` 与 `build_cluster_space` 定义原胞基元、对称性、相互作用截断和参数空间。
2. `Supercell` 与 `ClusterMap` 将原胞模型映射到一个明确给定的参考超胞。
3. `FiniteDifference` 重建单个阶次；`FitSystem` 构建并求解多个阶次的仅力拟合。

结构长度使用 Å，能量使用 eV，力使用 eV/Å。n 阶力常数的单位是 eV/Åⁿ。

## 选择工作流

- [有限差分](docs/zh/api/finite-difference-api.md)：生成有序位移结构，调用 ASE calculator，并重建一个阶次。
- [力拟合](docs/zh/api/fitting-api.md)：将带有已存储原子力的 ASE 结构流式写入可复用拟合系统，并联合求解多个阶次。
- [Q&A](docs/zh/Q&A.md)：了解力的存储、顺序、可辨识性、求解器和常见选择。

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

<!-- END GENERATED: docs/zh/index.md -->

## 文档

阅读[中文文档](https://gtiders.github.io/mlfcs/zh/)或[英文文档](https://gtiders.github.io/mlfcs/en/)。源文件分别位于 [docs/zh](docs/zh/) 和 [docs/en](docs/en/)。

## 引用

若 MLFCS 对发表工作有所贡献，请使用 [CITATION.cff](CITATION.cff) 中的软件引用信息。

## 贡献

参见 [CONTRIBUTING_ZH.md](CONTRIBUTING_ZH.md)和[问题追踪](https://github.com/gtiders/mlfcs/issues)。

## 许可证

MLFCS 按照 [GNU 通用公共许可证第 3 版或更高版本](LICENSE)发布。
