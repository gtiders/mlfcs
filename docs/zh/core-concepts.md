# 核心概念：从原胞模型到超胞数据

MLFCS 将**力常数模型**与**提供原子力的结构**分开。先理解三个对象：

| 对象 | 持有 | 不持有 |
| --- | --- | --- |
| `ClusterSpace`（`space`，常简称 CS） | 原胞、对称性、cluster、orbit，以及一个或多个阶次的独立力常数参数 | 超胞、位移结构或训练力 |
| `Supercell` | 明确给定的 ASE 超胞、它与原胞的整数晶格关系，以及每个原子对应的原胞位点 | cluster 或力常数参数 |
| `ClusterMap`（`mapping`） | 一个 CS 与一个超胞之间的联系：每个原胞 cluster 像对应哪些超胞原子 | 第二套 orbit 或拟合参数 |

构造顺序是 `原胞 ASE Atoms → ClusterSpace`、`明确给定的超胞 ASE Atoms → Supercell`，再由 `(ClusterSpace, Supercell) → ClusterMap`。有限差分和拟合使用 mapping；计算结果是绑定在原胞 CS 上的 `ForceConstants` 模型。

## ClusterSpace：原胞级模型

```python
from ase.build import bulk
from mlfcs import ClusterSpace

primitive_atoms = bulk("Si", "diamond", a=5.43)
space = ClusterSpace(
    primitive_atoms,
    cutoffs={2: 4.0, 3: 3.0},
    max_body_orders={2: 2, 3: 3},
    symprec=1e-5,  # Å；也是默认值
)

print(space.orders, space.n_parameters)
```

`ClusterSpace` 自行构造并持有 `space.primitive`。它以 `symprec` 识别原胞对称性，将等价 cluster 归入 orbit；每个 orbit 提供一组独立的笛卡尔张量参数。例子同时定义 FC2 和 FC3，各阶分别有截断和 cluster 中最多不同原子位点数（body order）。

正截断值表示 Å 单位的半径；负整数（如 `-7`）表示第七近邻壳层，由 CS 换算成半径。截断决定**模型本身**，改变截断就改变参数空间。CS 不持有超胞大小。

## Supercell：明确给定的原子结构

```python
from mlfcs import Supercell

supercell_atoms = primitive_atoms.repeat((3, 3, 3))
supercell = Supercell.from_atoms(space.primitive, supercell_atoms)

print(supercell.matrix)
print(len(supercell.numbers))
```

MLFCS 不会自动选择或扩展超胞。用户提供带有确定原子顺序的 ASE `Atoms`。按 ASE 的行向量约定，`Supercell.from_atoms` 找到整数晶格关系 $C_{\mathrm{s}} = M C_{\mathrm{p}}$，其中 `supercell.matrix` 就是 $M$。它还用 `space.primitive.symprec`（单位 Å）检查晶格，并把每个超胞原子唯一映射到原胞位点和周期平移。不需要另传 $M$。

可以映射不同顺序的等价超胞，但**训练结构必须保持所传超胞的具体原子顺序**。超胞的周期陪集标签标识落在同一个超胞原子上的原胞平移。

## ClusterMap：桥梁与秩检查

```python
from mlfcs import ClusterMap

mapping = ClusterMap.build(space, supercell)
for order in space.orders:
    info = mapping.rank_info(order)
    print(order, info.rank, info.parameters, info.nullity)
    info.require_full()
```

mapping 将原胞 orbit 的每个像折叠为超胞原子索引，不在超胞内重新定义 orbit。过小或形状不合适的超胞会让不同原胞相互作用落到相同观测上。`rank_info(order)` 在昂贵的力计算前检查这种**结构可辨识性**；若参数无法区分，`require_full()` 抛出 `AliasingError`。结构满秩仍不保证某一批拟合位移足够多样。

同一个 CS 可以配不同超胞；为每个超胞分别构建 `Supercell` 和 `ClusterMap`。取得 mapping 后可继续阅读[有限差分](finite-difference-api.md)或[力拟合](fitting-api.md)。
