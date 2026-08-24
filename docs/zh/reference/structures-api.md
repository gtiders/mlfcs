---
title: 结构、超胞与对齐 API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 结构、超胞与对齐 API

## `build_supercell`

```python
build_supercell(
    primitive: Atoms,
    supercell_matrix: object,
    *,
    symprec: float = 1e-5,
) -> Atoms
```

从显式 primitive 生成 ASE `Atoms` 超胞，原子顺序与 phonopy `is_old_style=True` 一致。该函数只负责结构
生成；拟合、有限差分、SCPH 和 SSCHA 不会接收扩胞矩阵或隐式调用它。

| 参数 | 含义 |
|---|---|
| `primitive` | 三维周期 ASE `Atoms`。坐标、元素和质量被复制到目标胞。 |
| `supercell_matrix` | 长度为 3 的整数 repeats，或非奇异整数 $3\times3$ 矩阵。内部统一采用 row-vector convention。 |
| `symprec` | 去重及 phonopy 构造容差，单位 Å。 |

返回新的周期 `Atoms`。若安装 phonopy，直接调用其 old-style 构造；否则使用项目内等价实现。非整数、奇异
矩阵、非周期 primitive 或非正 determinant 的 phonopy ordering 会抛出 `ValueError`。

```python
primitive = read("primitive.vasp")
reference = build_supercell(primitive, (4, 4, 4))
```

## `align_structures`

高级导入：

```python
from mlfcs.structure.relation import align_structures

align_structures(
    reference: Atoms,
    atoms: Atoms,
    *,
    tolerance: float = 1e-5,
) -> tuple[Atoms, float]
```

显式把 `atoms` 重排到 `reference` 原子顺序并返回最大匹配残差。它用于整理外部数据，不应隐藏在拟合或
有限差分内部。两结构必须拥有相同晶格、PBC、原子数和元素多重集；匹配超过 `tolerance` 时拒绝。

## `StructureRelation`

```python
from mlfcs.structure.relation import StructureRelation

StructureRelation.from_atoms(
    primitive: Atoms,
    reference: Atoms,
    *,
    tolerance: float = 1e-5,
) -> StructureRelation
```

该对象验证 reference 是 primitive 的整数超胞，并保存：

- `primitive`、`reference`；
- 整数 `supercell_matrix`；
- HNF-backed `PeriodicIndex`；
- reference 原子与 `(primitive_site, translation)` 的双射。

普通用户通常不直接构造它；`ForceConstants.relation`、realization、采样和 writer 会复用这一关系。

## 原子顺序规则

`reference` 是所有位移、力数组和稠密 IFC 的权威标签。MLFCS 不假设“同一晶格即可”，也不允许拟合器
静默交换原子。外部软件提供的结构若顺序不同，应先对齐，再把对齐后的结构和力共同保存。
