---
title: 结构、超胞与对齐 API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 结构、超胞与对齐 API

## `mlfcs.tools.supercell.build_supercell`

```python
from mlfcs.tools.supercell import build_supercell

build_supercell(
    primitive: Atoms,
    supercell_matrix: object,
    *,
    symprec: float = 1e-5,
) -> Atoms
```

从显式 primitive 生成 ASE `Atoms` 超胞，使用项目自身的 NumPy/ASE 实现并采用 primitive-site-major
原子顺序，不在运行时导入或调用 phonopy。该函数只负责结构生成；拟合、有限差分、SCPH 和 SSCHA
不会接收扩胞矩阵或隐式调用它。

| 参数 | 含义 |
|---|---|
| `primitive` | 三维周期 ASE `Atoms`。坐标、元素和质量被复制到目标胞。 |
| `supercell_matrix` | 长度为 3 的整数 repeats，或非奇异整数 $3\times3$ 矩阵。内部统一采用 row-vector convention。**不接受浮点形式**（如 `[[2.0, ...]]`）：扩胞矩阵是离散构造参数，不是可以四舍五入的近似量。 |
| `symprec` | 对包围框边界上的重复生成位置进行去重的长度精度，单位 Å。 |

返回新的周期 `Atoms`。非整数、奇异矩阵、非周期 primitive 或非正 determinant 会抛出 `ValueError`。

`mlfcs.tools` 是叶子便利包：它可以依赖 `mlfcs.structure`，但主线计算包不得导入它。主线入口不会为你
构造超胞，也不会在缺少 `reference` 时猜测一个；构造超胞与传入 `reference` 是两个显式步骤。

```python
primitive = read("primitive.vasp")
reference = build_supercell(primitive, (4, 4, 4))
```

## `mlfcs.tools.supercell.align_structures`

```python
from mlfcs.tools.supercell import align_structures

align_structures(
    reference: Atoms,
    atoms: Atoms,
    *,
    tolerance: float,
) -> tuple[Atoms, float]
```

显式把 `atoms` 重排到 `reference` 原子顺序并返回胞与原子匹配中的最大残差。它整理的是外部程序或 MD 产生的结构，
属于**外部导入策略**：`tolerance` 没有默认值，必须由调用方给出，而且它不参与核心结构身份判定。主线计算
路径不调用它——拟合与有限差分不会静默重排训练帧。两结构必须拥有相同晶格、PBC、原子数和元素多重集；
匹配超过 `tolerance` 时拒绝。

## `StructureRelation`

```python
from mlfcs.structure.relation import StructureRelation

StructureRelation.from_atoms(
    primitive: Atoms,
    reference: Atoms,
    *,
    symprec: float,
) -> StructureRelation
```

该对象验证 reference 是 primitive 的整数超胞，并保存：

- `primitive`、`reference`；
- 整数 `supercell_matrix`；
- HNF-backed `PeriodicIndex`；
- reference 原子与 `(primitive_site, translation)` 的双射；
- `symprec`：本次判定使用的唯一长度精度，单位 Å；
- `cell_residual`：每个原胞晶格系数的最大晶格残差，单位 Å；
- `position_residual`：原子映射的最大残差，单位 Å。

`tolerance` 已改名为 `symprec`，不保留别名。两处几何判定都使用严格小于：晶格残差与原子映射残差都必须
`< symprec`；超限时异常给出实测残差、`symprec`（单位 Å）与候选整数矩阵。晶格残差按 $\sum_j|S_{ij}|$
归一化，因此同一个 `symprec` 对 $1\times1\times1$ 与大重复矩阵含义一致——这不是第二个阈值。

普通用户通常不直接构造它；`ForceConstants.relation`、realization、采样和 writer 会复用这一关系。

## 原子顺序规则

`reference` 是所有位移、力数组和稠密 IFC 的权威标签。MLFCS 不假设“同一晶格即可”，也不允许拟合器
静默交换原子。外部软件提供的结构若顺序不同，应先对齐，再把对齐后的结构和力共同保存。
