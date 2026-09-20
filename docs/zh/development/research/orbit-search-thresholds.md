---
title: Orbit 搜索阶段的阈值与精确化
audience:
  - advanced
  - developer
status: research
code_verified: 4.0.0a6
---

# Orbit 搜索阶段的阈值与精确化

本笔记审计正式拟合或有限差分之前运行的全部代码：晶格关系、primitive orbit 搜索，以及把
primitive orbit 落实到有限 reference 中的 realization。它列出该路径上的数值阈值，以及那些
本可以用精确代数完成、目前却用浮点或定宽整数做的判定。

## 阶段图

```
Atoms
 ├─ StructureRelation.from_atoms                 structure/relation.py            tolerance 1e-5
 ├─ normalize_supercell_matrix                   structure/integer_lattice.py     atol 1e-10, int64 守卫
 ├─ PeriodicIndex                                structure/supercell_mapping.py   int64 余数键
 ├─ PrimitiveSymmetryOperations.from_atoms        structure/symmetry.py           spglib symprec
 └─ SymmetryOperations.from_primitive_operations  整数模运算
        ↓
 InteractionSpace.from_frame                     interactions/space.py           symprec 透传
        ↓
 build_primitive_interaction_space               interactions/primitive/builder.py      分数坐标整数
 ├─ traverse_indexed_orbit                        interactions/algebra/indexed_orbit.py  精确稳定子
 ├─ invariant_kernel                              interactions/algebra/invariants.py     模秩 + 验证过的核
 ├─ select_independent_rows                       interactions/algebra/invariants.py     QR 搜索，计数受认证
 └─ normalize_pivot_basis                         interactions/algebra/invariants.py     笛卡尔像上的浮点求解
        ↓
 realize_interaction_space +
 validate_realization_identifiability             interactions/realization.py      精确整数秩
        ↓
 ReciprocalQuotientGrid                           structure/reciprocal.py          atol 1e-12
```

## 该阶段的阈值清单

| 位置 | 值 | 决定什么 | 性质 |
| --- | --- | --- | --- |
| `space.py`、`symmetry.py` | `symprec=1e-5` $(\text{Å})$ | spglib 对称性精度 | 真阈值，保留 |
| `space.py` | 同一个 `symprec` | 又被当作**原子映射容差** | 一个数两个语义 |
| `relation.py` | `1e-5` | 整数超胞判定与原子映射代价 | 用浮点判整数事实 |
| `relation.py` | `1e-7` | 训练构型 cell 检查 | 同一问题紧 100 倍 |
| `relation.py` | `1e-5` | `align_structures` | 第三份同值 |
| `integer_lattice.py` | `atol=1e-10` | 「supercell_matrix 必须为整数」 | 用浮点判整数事实 |
| `reciprocal.py` | `atol=1e-12` | q 点兼容性复核 | 对精确标签做浮点复核 |
| `symmetry.py` | `symprec * 10.0` | 按距离匹配整数 site shift | 凭空松弛因子 |
| `candidates.py` | `+1e-8` | 邻居列表半径外扩 | 永不生效 |
| `invariants.py` | 相对 `1e-9` | 笛卡尔像上的 pivot 行搜索 | 仅搜索：pivot 个数由分数坐标的精确维数认证 |

## 本可精确代数完成的判定

| 位置 | 现在 | 精确路线 |
| --- | --- | --- |
| `invariant_kernel` | 大素数模秩 + 整数验证核 | 已完成：约束行 = 整数 spglib 旋转作用于 $0/1$ label 基，维数 $C - \operatorname{rank}_{\mathbb{F}_p}(G)$，返回列用 `rows @ basis == 0` 验证 |
| `select_independent_rows` | QR 搜索，计数由分数坐标维数认证 | 已完成：没有任何结论依赖该阈值 |
| `normalize_pivot_basis` | `np.linalg.solve` | 仍是浮点，但作用在精确整数基的确定性笛卡尔像上 |
| `indexed_orbit._action_signature` | 整数分数坐标元组键 | 已完成 |
| `indexed_orbit` 稳定子残差 | 整数稳定子作用，无过滤 | 已完成 |
| `validate_realization_identifiability` | 整数分数基 + 整数作用矩阵，精确秩（模秩快路 + 精确兜底） | 已完成 |
| `StructureRelation.from_atoms` | `allclose(transform, matrix)`、LAP 映射代价 | 有理基变换，再按折叠后的分数坐标做精确陪集匹配 |
| `reciprocal.py` | `allclose(points @ matrix.T, rint(...))` | 整数标签上的模恒等式 |
| `integer_lattice.py` | `allclose(values, rint(values), atol=1e-10)` | 由调用方精确导出整数矩阵后，整数性即整除判定 |

## 定宽整数

`int64` 本身不是问题：模运算在 `int64` 中是精确的。代价出现在把精确代数结果塞进定宽数组时，
由此被迫增加本不需要的守卫：`structure/integer_lattice.py` 里三处 `np.iinfo(np.int64)` 溢出检查
与 `_sympy_int64_matrix` 转换，只是为了把精确的 `sympy` 整数搬进 `int64` 数组。若计算全程留在
`sympy`（或 Python `int`）中，直到最后对**有界输出**（labels、translations、representatives）做
数值化，这些守卫与转换即可消失。

`interactions/keys.py`、`structure/supercell_mapping.py`、`structure/reciprocal.py` 中的标签、
平移与余数键确实是整数，应当保持整数；需要改变的是它们的**来源**——应当是精确整数运算，而不是
四舍五入后的浮点。

## 前两项修复的实测效果

基准：Ba8Ga16Ge30 参考（54 原子原胞、$2\times2\times2$ 超胞），近邻 shell 取 $-8$（FC2）、
$-8$（FC3）、$-5$（FC4），body 为 2/3/4，`symprec=1e-4`。解析出的半径为 4.8746、4.8746、
4.0252 Å，orbit 数 213、938、628，参数数 1791、21270、26628。

| 阶段版本 | realization | 整段 |
| --- | --- | --- |
| 容差过滤 + `matrix_rank`（改前） | 25.03 s | 28.51 s |
| 精确版，每个 basis entry 一次 `Fraction` | 40.07 s | 43.63 s |
| 精确版，整数流水线（当前） | 22.64 s | 26.23 s |

按阶看 realization：改前 0.47 / 6.57 / 19.17 s，当前 0.44 / 6.26 / 17.54 s。

差异来自三点：basis 只按其少数几个不同幅值做一次精确重构（而不是每个 entry 一次）；tensor action
矩阵按精确整数签名缓存（而不是每张 image 重建）；系数累加与分量 rank 全程整数运算，只在 rank 时
按分量给出一个公共分母。

## 建议顺序

1. `realization.py`：精确 rank 与精确零判定。`> 1e-10` 过滤可能静默丢掉非零系数并翻转
   可辨识性结论，因此这首先是正确性修复。
2. `indexed_orbit._action_signature`：用整数键取代四舍五入浮点。改动自洽、风险最低。
3. `invariants.py`：精确 kernel 与 rank，需要先消去 label 对称基的 $1/\sqrt{k}$ 缩放。完成后
   orbit 搜索入口的 `tolerance` 参数可以删除。
4. `relation.py`：有理超胞判定与精确陪集匹配，从而从 `StructureRelation.from_atoms` 移除映射
   容差，把 `symprec` 只留给 spglib。

该阶段真正的阈值只有 `symprec`（spglib）与邻居 cutoff 半径，二者应当保留。

## 分数坐标整数代数（hiphive 1.5）

第 3 项原先写在笛卡尔坐标系里。与参考坐标轴不对齐的原胞（fcc 原胞 $60^\circ$、六方、三方）
其作用矩阵含无理项（$\sqrt{3}/2$、$1/\sqrt{3}$），轨道基因此没有精确有理形式，
`realization` 需要有理重构，重构失败时还得退回浮点路径。

hiphive 从结构上避开该问题：对称性代数全部在**分数（scaled）坐标系**中完成。

| hiphive | 作用 |
| --- | --- |
| `cluster_space.py:164` | `rotation_matrices` 即 spglib `dataset.rotations`，分数坐标下的整数矩阵 |
| `core/eigentensors.py:26` | 约束矩阵由这些整数旋转作用于 $0/1$ label 对称指示张量得到，条目全为精确整数 |
| `core/utilities.py:25` | `SparseMatrix.rref_sparse` / `nullspace` 用 sympy 精确有理消元求核 |
| `core/eigentensors.py:112` | `renormalize_to_integer` 把解向量乘分母最小公倍数，得到整数 eigentensor |
| `core/tensors.py:38` | `rotation_to_cart_coord(R, cell)`（$\text{cell}^T R \,\text{cell}^{-T}$）与 `rotation_tensor_as_matrix` 只在边界转笛卡尔 |
| `core/config.py:33` | 唯一记录的例外（`eigentensor_simplify_before_compress`）只针对 hcp 这类笛卡尔旋转 |

照搬其符号实现不可行：用它的 `SparseMatrix` 核按整个点群逐轨道迭代，四阶每轨道耗时
70.7 ms（立方，48 个操作）、55.2 ms（fcc 原胞）、30.4 ms（六方），对应我们 448–628 轨道的
四阶空间需要 13–44 s，而现 `invariants` 阶段只要 0.096 s。

框架洞察可以迁移，代价换成快速整数代数：

- 实测所有晶胞（立方、六方、fcc 原胞 $60^\circ$）的 spglib 分数旋转都是精确整数，笛卡尔的不是；
- 秩的**两个方向**都由模运算证明：某素数下满秩即为结论；判定亏秩时持续加入素数，直到其乘积超过最大
  子式的 Hadamard 上界（用精确 `isqrt` 行范数；浮点上界会损失精度、进而给出假证书），此时必有一个素数
  是好的；素数是**无上限的确定性流**（元素达 $2^{31}$ 的矩阵需要几十个素数）。真正亏秩的矩阵无论加多少
  素数都保持其低秩（模秩永不超精确秩）。原先用于判定亏秩的分数自由消元已删除：为整个四阶空间的 12 个
  Gram 出证书耗 378 ms，被拒绝的参考胞现在 1–48 ms 内报出（旧消元对单个 27×27 Gram 需要 14 s）
- 整数作用在 `int64` Kronecker 缩并下精确，约束块为 $3^{\text{order}} \times C$；核维数是
  $C - \operatorname{rank}_\mathbb{Q}(G)$，$G = B^T B$，而模大素数（$2^{31}-1$）的秩在这里可以
  定论，因为 $\operatorname{rank}_{\mathbb{F}_p} \le \operatorname{rank}_\mathbb{Q}$：通过即证明，
  不通过可检测；
- 先化到 Gram（此处 $54 \times 54$）再求秩可把精确秩压到每轨道 1 ms 以内，直接对 3448 行约束栈
  取模秩则要 100 ms；
- 真正的成本驱动量是每轨道**不同稳定子**的个数，实测均值 1.3–2.9（最大 19），而非点群的 48 个
  操作，因此精确路线只增加约 1–2 s（448 轨道四阶空间）。

已实现（形式受下游约束所限）。两框架标定得到 $K = (\text{cell}^T)^{\otimes\,\text{order}}$，配
`symmetry.rotations[operation]`（不转置，按 $\text{after} \cdot \text{before}$ 复合），与 hiphive 的
`rotation_to_cart_coord` 是同一个映射；测试逐生成元与 spglib 自带笛卡尔旋转对照钉住。

- `PrimitiveInteractionOrbit.basis` 是**分数坐标下不变子空间的精确整数基**，一列一个拟合参数，
  参数即该基的系数。**不做规范形**：整数子空间的 pivot 块一般是有理的，要求它为单位阵会在**任何**
  框架下都强制有理参数化。
- `invariant_kernel` 返回前把每列除以自身元素最大公约数；`realization` 用同一基对分数坐标实现
  矩阵求精确秩。
- 消费者只做**一次**确定性映射 $K$ 得到物理张量：`fitting/parameterization.py`、
  `force_constants/expansion.py`、`constraints/translational.py`、`finite_difference/reconstruction.py`
  统一读 `space.cell` 与 `orbit.basis`。

数值已对照改前代码验证：有限差分重建的力常数相对偏差 $2.3\times10^{-16}$，拟合预测力**逐位一致**；
只有参数坐标因换基而改变（同一子空间的另一组系数）。

### 为什么观测分量仍在笛卡尔侧选

有限差分计划观测的是**笛卡尔**张量分量，且必须观测 $\dim$ 个"其行能确定参数"的分量。这个独立性是
笛卡尔像上的**代数条件**而非整数条件，没有符号运算就无法精确判定。在 4 个体系、2/3 阶上，若改在
分数坐标用精确整数贪心秩选这些行，再在这些行处对笛卡尔块求逆：

| 体系 | 行集合相同 | 分数行处笛卡尔块的最大条件数 |
| --- | --- | --- |
| 立方简胞 | 3/3, 2/2 | 1.0, 1.0 |
| fcc 原胞 | 4/4, 4/10 | 5.7, $\infty$（奇异） |
| 六方 | 5/5, 12/12 | 4.6, 9.6 |
| SnSe | 6/6, 4/4 | 7.6, 7.6 |

fcc 原胞三阶就已出现奇异块——那会让观测计划**信息不足**（不只是病态）。因此该选择留在笛卡尔像上，
其**条数**由精确分数维数认证；这也是本阶段仅剩的一处数值秩判定。

## `src/` 全量阈值清单

同一套审计覆盖包内其余部分，按"这个数字决定了什么"分组。**不随数据缩放**的那些才是脆弱的。

### 本可精确代数完成的判定（用浮点判整数/有理事实）

| 位置 | 值 | 背后的精确命题 |
| --- | --- | --- |
| `structure/integer_lattice.py:24` | `atol=1e-10` | 超胞矩阵各元素为整数 |
| `structure/relation.py:57` | `atol=tolerance`（即 `symprec` 值） | 同一命题上一层 |
| `structure/relation.py:82` | `>= tolerance` | LAP 原子映射代价为零，即精确陪集匹配 |
| `structure/relation.py:110` | `atol=1e-7` | 训练构型 cell 与参考一致 |
| `structure/relation.py:132,148` | `tolerance=1e-5` | 对齐后的结构与参考一致 |
| `structure/reciprocal.py:40` | `atol=1e-12` | q 点标签是整数组合 |
| `structure/symmetry.py:92` | `symprec * 10.0` | 整数 site shift 匹配，松弛因子凭空而来 |
| `structure/periodic_geometry.py:103,158` | `atol=1e-8 + rtol*max(...)` | 简并最小像（相对量，保留） |
| `structure/periodic_geometry.py:171,176` | `rtol=1e-5, atol=1e-8` | 去重后的邻居距离，决定截断壳层 —— **已标记待重写，见下** |
| `interactions/primitive/candidates.py` | 已移除 | 两处启发式只存在于 `cutoff=None` 分支，该分支已删除；现在必须给距离或负壳层序号 |
| `force_constants/realization.py:64,94` | `atol=1e-7`、`> 1e-5` | 目标与源是同一晶格、同一原子集合 |
| `io/alamode.py:64,192`、`io/shengbte.py:55` | `1e-10`、`1e-8/1e-10` | 导出内容为整数或重复时的一致性校验 |

### 会改变模型的判定（最高优先）

| 位置 | 值 | 决定什么 |
| --- | --- | --- |
| `fitting/linear_solvers.py:51` | `tolerance * max(block.shape) * max(diagonal)` | 约束零空间中的**自由参数个数** |
| `fitting/constraints.py:66,74` | `1e-12`、`np.round(data, 12)` | **保留**：行是代数量在笛卡尔分量下的求值，任何整数键都无法判定相等；余量见下 |
| `constraints/translational.py:36` | `abs(entry) > 1e-12` | **保留**：同一把守卫；行尺度 59 时代数零落在 $3.5\times10^{-15}$，真实系数是 $O(10)$ |
| `fitting/gram/models.py:38` | `max(norm) * 1e-12` | 哪些 Gram 列被归一化 |
| `interactions/algebra/invariants.py:30` | 相对 `1e-9` | 观测分量行；**条数**由精确维数认证 |

### 应当保留的阈值

- 真物理阈值：`symprec`（spglib）、邻居 cutoff 半径、有限差分 `displacement`、THz 量纲的
  `imaginary_tolerance` 与 `cutoff_frequency`、`mixing`、旋转和规则的 `strength`。
- 求解器收敛容差（应保持"容差"语义）：`Fitter.tolerance`、`explicit_constraint_null_space` 的容差、
  `fitting/linear_solvers.py` 中 CG/LSMR 的 `rtol`/`atol`、`constraints/translational.py` 的 ASR 投影
  容差、`SCPH.tolerance`（THz），以及 `constraints/rotational.py` 与 `fitting/gram/models.py` 中的
  相对谱截断。
- 文本导出的零化（决定文件长什么样，而非模型是什么）：`io/numeric_text.py::zero_small_scalar`
  及 `io/alamode.py`、`io/shengbte.py` 的 `_TEXT_ZERO_TOLERANCE`，以及 `_MIRROR_TOLERANCE_BOHR`。

### 标记：`unique_periodic_distances` 用浮点守卫决定壳层

`unique_periodic_distances(rtol=1e-5, atol=1e-8)` 是唯一把邻居距离变成"壳层"的地方，且只在调用方
传入负 `cutoff`（壳层序号）时由 `candidates.py:72` 调用。它的去重是对"格矢整数组合"的**浮点**判定，
因此它给出的壳层边界继承了这把尚未像本笔记其他阈值那样被论证过的守卫。重写时应二选一：从种子与晶格
数据**结构性地**导出壳层（不再依赖距离），或把实测余量写进文档。此处**仅标记，不改行为**。

### 谁在负责可辨识性

真正能拒绝坏选择的是 realization 检查。拟合与有限差分两条路径都经由
`InteractionSpace.realized_orbit_space`（`interactions/space.py:164`）到达 `realize_interaction_space`，
进而调用 `validate_realization_identifiability`（`interactions/realization.py:182`）。该函数在**精确
整数分数坐标**下对实现矩阵求秩，当某个连通参数分量秩亏时抛 `InteractionAliasingError`，并指出冲突
团簇与两条补救办法（"更大的单一参考超胞或更短的截断"）。

`cutoff=None`（原以"把半径收到实测边界以内"来**预防**混叠）已被移除，职责交给上述**执法**，它更强：
精确（整数秩、无阈值）、覆盖全部消费者、并且**报错而非静默折叠**。调用方现在必须给出距离或负的壳层
序号；各教程把"该参考胞解析出的半径"记为显式数值（取自它们自己提交的 metadata）。它不覆盖的：半径
是否物理合理，以及 `StructureRelation` 自身使用的映射容差。

### 为什么 ASR 系数守卫保留

曾尝试用每行的整数像替换去重键并删掉系数守卫，两项都被实测否决：一条方程的笛卡尔行是
$e_c^\top M\,(S_{\text{lat}}B)$，而按同一分量下标构造的整数行是 $e_c^\top (S_{\text{lat}}B)$，
两者并非彼此的像，因此那个键在比较**另一个对象**，可能把不等价的行合并；而失去守卫后，同一条物理
约束的各行支撑不再一致，去重只保留 18 条中的 13 条（原本 1 条），下游阈值化的秩判定随之把带约束
拟合移动了整整一倍。守卫本身也并不"尖锐"：它比上述噪底高约 300 倍，比最小真实系数低十个数量级。

### 两处较小的异味

`phonon/sampling/structures.py:57,59,69` 分别用 `cutoff_frequency != 0.01`、`imaginary_tolerance != 1e-6`、
`displacement != 0.01` 作为"调用方未改默认值"的哨兵；同一个"cell 是否一致"的问题在 `structure` 包里
出现了三个不同数值（`1e-5`、`1e-7`、`1e-10`）。
