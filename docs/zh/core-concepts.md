# 核心概念

MLFCS 使用 ASE 结构作为晶体和力数据的用户入口。核心模型对象是 `ClusterSpace`（常简称 CS）：它记录后续算法所需的原胞基元和对称性信息，并定义力常数簇及参数。明确给定的 `Supercell` 提供实际计算力的结构，`ClusterMap` 将模型连接到该超胞。

## `ClusterSpace`：定义原胞模型

`ClusterSpace` 是通常的起点。直接从描述原胞的 ASE `Atoms` 构造；它会创建并保存供后续对称性、晶格坐标、质量及映射使用的原胞表示。

```python
from ase.build import bulk
from mlfcs import ClusterSpace

si = bulk("Si", "diamond", a=5.43)
space = ClusterSpace(
    si,
    cutoffs={2: -7, 3: -6},
    max_body_orders={2: 2, 3: 3},
    symprec=1e-5,
)
print(space.orders, space.n_parameters)
```

公开构造函数为：

```python
ClusterSpace(atoms, *, cutoffs, max_body_orders, symprec=1e-5)
```

| 参数 | 含义 |
| --- | --- |
| `atoms` | 描述原胞基元的 ASE `Atoms`，位点顺序按用户所需指定。该对象中的质量会传入模型。 |
| `cutoffs` | 从力常数阶数映射到相互作用截断。正值是 Å 单位的半径；负整数选择近邻壳层，例如 `-7` 表示第七壳层。每个配置阶次都需提供一项。 |
| `max_body_orders` | 从力常数阶数映射到簇中不同位点数的最大值。每个配置阶次都需提供一项。 |
| `symprec` | 对称性和结构映射精度，单位 Å，默认 `1e-5`。 |

`ClusterSpace` 提供 `primitive`、`symmetry`、`blocks`、`orbits`、`orders`、`pbc`、`n_parameters`、`parameter_offsets` 和 `fingerprint`；`block(order)` 返回该阶的设置以及参数/orbit 切片。内部的 `space.primitive` 是供后续算法使用的只读原胞信息表示，不是用户需要额外执行的一步构造。质量从输入 ASE 结构继承；若需指定同位素或位点质量，应在创建模型前设置到 ASE `Atoms` 上。

正截断值表示 Å 单位的半径；负整数表示近邻壳层，例如 `-7` 表示第七壳层。各阶分别设置 cutoff 和 body order，因此谐性与非谐阶次可以同时定义。

## `Supercell`：提供力计算结构

`Supercell` 描述实际用于计算力的 ASE 结构。用户提供原子及其顺序；MLFCS 推断整数基变换关系，并确定每个超胞原子对应的原胞位点和平移。

```python
from mlfcs import Supercell

supercell_atoms = si.repeat((3, 3, 3))
supercell = Supercell.from_atoms(space.primitive, supercell_atoms)
print(supercell.matrix, supercell.determinant, len(supercell.numbers))
```

构造函数为 `Supercell.from_atoms(primitive, atoms)`：`primitive` 使用 `space.primitive`，`atoms` 是明确给定的 ASE 超胞。不需要另传矩阵。按照 ASE 行向量晶格约定，推断关系为 $C_s = M C_p$，`supercell.matrix` 即 $M$。

对象提供 `primitive`、`matrix`、`cell`、`scaled_positions`、`numbers`、`sites`、`translations`、`quotients` 和 `determinant`。这些数组遵循输入超胞的原子顺序。位移结构和力数据都应保持该顺序。原胞的 `symprec` 用于晶格与原子映射。

## `ClusterMap`：连接模型和数据几何

`ClusterMap` 将原胞簇像折叠到一个明确超胞的原子索引。为选定的 `ClusterSpace` 和 `Supercell` 构造一份 mapping：

```python
from mlfcs import ClusterMap

mapping = ClusterMap.build(space, supercell)
for order in space.orders:
    info = mapping.rank_info(order)
    print(order, info.rank, info.parameters, info.nullity, info.aliases)
    info.require_full()
```

构造入口为 `ClusterMap.build(space, supercell)`。mapping 提供 `space`、`supercell`、`atoms` 和 `fingerprint`；`atoms` 按 orbit 保存各簇像对应的超胞原子索引。`aliases(order)` 报告折叠到相同有序原子组的原胞像。

`rank_info(order=None)` 返回包含 `parameters`、`rank`、`aliases`、`nullity` 和 `full` 的 `RankInfo`。省略 `order` 时汇总所有配置阶次。若所选超胞无法区分该模型块中的全部参数，`require_full()` 会抛出 `AliasingError`。这项结构可辨识性检查适合在力计算前执行；它与某一训练集是否包含足够多样的位移是不同问题。

完整对象流程是 `原胞 ASE Atoms → ClusterSpace`、`超胞 ASE Atoms → Supercell`，再由 `(ClusterSpace, Supercell) → ClusterMap`。有限差分和拟合[共用该映射](finite-difference.md)。非谐后处理见 [SCPH](scph.md)，随机有限温度力匹配见 [SSCHA](sscha.md)。

## Si 示例

[Si 有限差分脚本](../../tutorial/SI/finite-difference-fc2/run.py)用简洁的 FC2 流程展示 `ClusterSpace`、明确给定的 `SPOSCAR` 和 `ClusterMap`。[Si 拟合脚本](../../tutorial/SI/fitting/fit.py)配置了多个阶次，并使用相同的模型到超胞映射方式。
