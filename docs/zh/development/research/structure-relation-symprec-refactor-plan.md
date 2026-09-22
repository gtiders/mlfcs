---
title: 原胞—超胞关系与单一 symprec 策略重构计划
audience:
  - advanced
  - developer
status: research
code_verified: 4.0.0a6
localized_only: true
---

# 原胞—超胞关系与单一 `symprec` 策略重构计划

## 0. 文档状态

- 目标分支：`dev`
- 实施性质：MLFCS 4.0 开发期破坏性重构，不保留旧参数别名，不提供兼容层。
- 目标：计算 API 强制接收用户提供的显式超胞；超胞构造移入叶子工具包；原胞对称性识别和原胞—超胞几何映射统一使用唯一的长度精度 `symprec`。
- 本文是实施规范。实现者如果发现本文与代码现状不一致，应先记录差异和最小复现，不得自行增加第二个映射精度。

## 1. 最终结论

本次重构只保留一个参与核心结构判定的用户参数：

```python
symprec: float  # 单位为 Å
```

它表达统一的物理问题：两个周期几何对象在笛卡尔空间中相差多少时，仍视为同一个对象。

`symprec` 用于：

1. spglib 的原胞对称性识别；
2. 原胞晶格与显式超胞晶格的整数复制关系验证；
3. 显式超胞原子到“原胞原子 + 整数格矢”的匹配；
4. 固定胞训练帧与参考超胞的胞一致性验证。

不得新增 `mapping_tolerance`、`cell_tolerance`、`position_tolerance` 或同义参数。不得在上述路径中写入 `1e-5`、`1e-7`、`1e-10` 等局部判定常数。

spglib 的 `angle_tolerance` 不是第二个长度精度。当前继续使用 spglib 的自动策略，即不传该参数或显式传 `-1.0`；不得把它用于原胞—超胞映射，也暂不暴露为 MLFCS 公共参数。

`align_structures` 若保留为处理外部 MD 帧或独立程序输出的模糊导入工具，可以有一个独立的 `tolerance`。它属于非核心导入策略，不参与 `StructureRelation`、拟合、有限差分、SCPH 或 SSCHA 的结构身份判定。

## 2. 架构边界

### 2.1 目标目录结构

```text
src/mlfcs/
├── structure/
│   ├── integer_lattice.py
│   ├── lattice_frame.py
│   ├── periodic_geometry.py
│   ├── relation.py
│   ├── supercell_mapping.py
│   └── symmetry.py
└── tools/
    ├── __init__.py
    ├── supercell.py
    └── structure_alignment.py    # 仅在决定迁移 align_structures 时建立
```

`structure` 是主线基础包；`tools` 是叶子便利包。依赖方向只能是：

```text
mlfcs.tools -> mlfcs.structure
```

下列生产包不得导入 `mlfcs.tools`：

- `structure`
- `interactions`
- `force_constants`
- `constraints`
- `finite_difference`
- `fitting`
- `phonon`
- `io`
- `calculators`

`tools` 可以依赖 ASE、NumPy、phonopy 和 `mlfcs.structure`，但核心代码不得为了获得参考超胞而调用 `tools`。测试和教程脚本可以由用户侧显式导入工具。

### 2.2 超胞必须由用户提供

以下主线入口必须继续或改为强制要求显式 `reference: Atoms`：

- `InteractionSpace`
- `FiniteDifferenceCalculation`
- `ForceConstantFitter`
- `SSCHA`
- 任何后续 SCPH/SSCHA 共享的结构关系入口

主线入口不得接受 `supercell_matrix` 后内部构造超胞，也不得在 `reference is None` 时自动构造。缺少 `reference` 应由 Python 签名直接报错；不要写隐式默认值，也不要根据 cutoff 猜测超胞。

用户若需要便利构造，显式调用：

```python
from mlfcs.tools.supercell import build_supercell

reference = build_supercell(primitive, supercell_matrix, symprec=1e-5)
calculation = FiniteDifferenceCalculation(
    primitive,
    reference=reference,
    order=3,
    cutoff=4.0,
    symprec=1e-5,
)
```

这是两个明确步骤，不允许计算入口反向调用第一步。

### 2.3 公共导出是有意破坏的

删除：

```python
from mlfcs import build_supercell
from mlfcs.structure import build_supercell
```

只保留：

```python
from mlfcs.tools.supercell import build_supercell
```

不得保留转发函数、弃用别名或 `__getattr__` 兼容。相应教程和测试一次性迁移。

## 3. 精度语义与数学定义

### 3.1 基本约定

- ASE 晶格矢量按行存储。
- 原胞晶格记为 $A_p$，参考超胞晶格记为 $A_s$。
- 超胞整数矩阵记为 $S$，满足 $A_s \approx S A_p$。
- `symprec` 必须是有限正浮点数，单位为 Å。
- 所有与 `symprec` 比较的量也必须是笛卡尔长度或“每个原胞晶格系数”的笛卡尔长度，单位为 Å。
- 禁止把无量纲矩阵元素、分数坐标差或角度直接和 `symprec` 比较。

### 3.2 候选整数矩阵

从用户提供的两个晶胞计算：

$$
T = A_s A_p^{-1}, \qquad S = \operatorname{rint}(T).
$$

`rint` 只负责生成待验证的候选整数矩阵，不构成接受判据。把 $T$ 与 $S$ 的无量纲元素差和 `symprec` 比较是错误实现。

候选 $S$ 转成整数后必须通过纯离散验证：

1. 形状是 $3 \times 3$；
2. 元素落在 `int64` 范围；
3. 行列式非零；
4. $|\det S| N_p = N_s$；
5. 各元素的化学计量满足超胞倍数关系。

这些都是整数事实，不使用任何浮点阈值。

### 3.3 晶格残差

重建晶格：

$$
\widehat{A}_s = S A_p, \qquad \Delta A = A_s - \widehat{A}_s.
$$

为避免大重复矩阵仅因长度规模而被更严苛对待，对第 $i$ 条超胞晶格矢量定义每个原胞晶格系数的残差：

$$
r_i^\mathrm{cell} =
\frac{\left\|\Delta A_i\right\|_2}
{\max\left(1, \sum_j |S_{ij}|\right)},
\qquad
r_\mathrm{cell} = \max_i r_i^\mathrm{cell}.
$$

接受条件统一为：

$$
r_\mathrm{cell} < \mathrm{symprec}.
$$

这里的归一化不是第二个阈值；它只把晶格误差表达为 Å/原胞晶格系数，使相同的 `symprec` 对 $1 \times 1 \times 1$ 和大超胞具有一致含义。

实现者必须用至少一个大重复矩阵测试验证该定义。若实测证明不归一化的绝对晶格残差更符合 spglib 的实际判据，应在独立提交中给出反例、两种定义的数值和教程影响，再请求决定；不得静默改公式或新增 `cell_tolerance`。

### 3.4 原子映射残差

保留按元素分组的全局线性指派，不改成精确分数坐标哈希，也不使用逐原子贪心。

对每种元素，候选槽位由原胞站点 $a$ 和整数陪集平移 $t$ 组成：

$$
r_{a,t} = r_a^p + t A_p.
$$

使用参考超胞的精确最小镜像算法计算参考原子 $i$ 到槽位 $(a,t)$ 的笛卡尔距离：

$$
d_{i,(a,t)} =
\left\|\operatorname{MIC}_{A_s}
\left(r_i^s-r_{a,t}\right)\right\|_2.
$$

用匈牙利算法求该元素的全局最小指派，定义：

$$
r_\mathrm{atom}=\max_i d_{i,\pi(i)}.
$$

接受条件统一为：

$$
r_\mathrm{atom}<\mathrm{symprec}.
$$

边界约定必须统一为严格小于；不要在不同路径混用 `<`、`<=`、`>=`。测试必须包含恰好位于边界、略低于边界和略高于边界三个案例。

保留 `PeriodicIndex` 的全局“每个原胞站点、每个陪集恰好出现一次”离散校验。不要在 `relation.py` 重写第二套同类校验。

### 3.5 固定胞训练帧

`StructureRelation.displacement(atoms)` 处理的是固定胞训练帧。删除当前隐藏的 `atol=1e-7`，使用建立关系时保存的 `self.symprec`。

训练胞残差定义为对应晶格矢量差的最大二范数：

$$
r_\mathrm{frame}=\max_i
\left\|A_i^\mathrm{frame}-A_i^\mathrm{reference}\right\|_2.
$$

接受条件为 $r_\mathrm{frame}<\mathrm{symprec}$。训练原子的真实位移不与 `symprec` 比较；它正是 `displacement` 要返回的物理量。这里只验证原子数、元素顺序和固定胞身份。

若未来支持变胞训练，那是新的物理模型和 API，不得通过放宽此处阈值偷偷实现。

## 4. 阈值数量与源码出现位置

### 4.1 用户可配置阈值

核心原胞—超胞关系中只有一个：

| 参数 | 单位 | 所属策略 |
|---|---:|---|
| `symprec` | Å | 对称性识别、晶格整数关系、原子映射、固定胞身份 |

非核心工具最多还有一个：

| 参数 | 单位 | 所属策略 |
|---|---:|---|
| `tolerance` | Å | `mlfcs.tools.structure_alignment` 对外部结构的模糊对齐 |

`angle_tolerance=-1.0` 是 spglib 的自动角度策略，不计为 MLFCS 用户可配置阈值。

### 4.2 核心映射中 `symprec` 的判定出现次数

按“实际发生阈值比较的代码位置”计数，`StructureRelation` 路径应当正好有 **3 处**：

1. `relation.py`：`r_cell < symprec`；
2. `relation.py`：`r_atom < symprec`；
3. `relation.py`：`r_frame < self.symprec`。

另有 **1 处参数转交**，不重复做本地阈值判断：

4. `symmetry.py`：`spglib.get_symmetry_dataset(..., symprec=symprec)`。

`LatticeFrame` 内现有的碰撞检查同样复用 `symprec`，属于规范化原胞的防歧义检查，不得另设数值。实施报告应把它单列，不要把它误报成新的用户阈值。

`build_supercell` 工具中有 **1 处工具内判定或参数转交**：传给 phonopy；无 phonopy 时 fallback 用同一个 `symprec` 去重。它不属于核心调用链，也不得被主线依赖。

若迁移 `align_structures`，工具内允许 `tolerance` 出现两种比较角色：胞残差和原子匹配残差。二者仍是同一个外部导入策略参数。

### 4.3 必须删除的隐藏值

当前入口层以下判定必须消失：

| 当前位置 | 当前值 | 处理 |
|---|---:|---|
| `structure/relation.py` 的关系默认参数 | `tolerance=1e-5` | 参数改名为 `symprec` |
| `structure/relation.py` 的胞矩阵 `allclose` | 复用 `tolerance` 但比较无量纲矩阵 | 改为第 3.3 节的 Å 残差 |
| `structure/relation.py` 的原子映射 | `tolerance` | 改为同一个 `symprec` |
| `StructureRelation.displacement` | `atol=1e-7` | 改为 `self.symprec` 与 Å 残差 |
| `structure/integer_lattice.py` 浮点取整 | `atol=1e-10` | 删除浮点输入分支，函数只接受整数 |

## 5. 分阶段实施

### P0：基线与失败测试

在改生产代码前完成：

1. 记录 `git status --short`，不得提交或覆盖现有无关修改。
2. 运行并记录：

   ```bash
   pytest -q \
     tests/test_core_structure_relation.py \
     tests/test_core_supercell.py \
     tests/test_core_supercell_builder.py \
     tests/test_architecture_dependencies.py
   ```

3. 新增先失败的契约测试：
   - `from_atoms(..., tolerance=...)` 抛出指名 `tolerance` 的 `TypeError`；
   - `StructureRelation.from_atoms(..., symprec=...)` 接受合法显式超胞；
   - 缺少显式 `reference` 的各计算入口失败；
   - `from mlfcs import build_supercell` 失败；
   - `from mlfcs.structure import build_supercell` 失败；
   - `from mlfcs.tools.supercell import build_supercell` 成功；
   - 主线包没有任何 `mlfcs.tools` 导入。

4. 增加 AST 门禁，禁止在以下文件的几何判定中出现浮点容差字面量：
   - `structure/relation.py`
   - `structure/integer_lattice.py`
   - `structure/supercell_mapping.py`

AST 测试应针对比较表达式和 `allclose`/`isclose` 的 `atol`、`rtol`，不要粗暴禁止所有浮点字面量，以免把日志格式或非阈值物理输入误报。

### P1：把超胞构造移到 `tools`

1. 新建 `src/mlfcs/tools/__init__.py`，保持最小内容；不要从中再次顶层导出 `build_supercell`，让导入路径明确。
2. 将 `src/mlfcs/structure/supercell.py` 移为 `src/mlfcs/tools/supercell.py`。
3. 工具继续接受显式整数 `supercell_matrix` 和 `symprec`，继续保持 phonopy old-style 顺序。
4. 删除 `mlfcs.__init__` 和 `mlfcs.structure.__init__` 中的 `build_supercell` 导出。
5. 将工具测试移名为 `tests/test_tools_supercell.py`，测试 phonopy 与 fallback 一致。
6. 教程脚本若确实需要构造超胞，改为从 `mlfcs.tools.supercell` 显式导入。
7. 生产代码不得因移动而新增任何工具导入。

工具自身的矩阵输入必须是整数数组、整数三元组或元素均为 Python/NumPy 整数的嵌套序列。浮点形式的 `[[2.0, ...]]` 直接拒绝；这不是数值近似问题，而是工具调用方声明离散构造参数的问题。

### P2：收紧整数工具的离散职责

修改 `normalize_supercell_matrix`：

1. 只接受整数 dtype 或逐项为整数对象的输入；
2. 删除 `np.rint` + `np.allclose(..., atol=1e-10)` 的浮点回退；
3. 保留形状、`int64` 范围和非奇异性验证；
4. 所有调用者传入已经确定的整数矩阵。

`StructureRelation.from_atoms` 是唯一允许从两个浮点晶胞生成候选 $S$ 的位置。它先 `rint`，再显式转为整数，最后按第 3 节做物理验证。不得让通用整数工具重新猜测浮点输入的意图。

### P3：重写 `StructureRelation` 的单一精度契约

目标签名：

```python
@classmethod
def from_atoms(
    cls,
    primitive: Atoms,
    reference: Atoms,
    *,
    symprec: float,
) -> StructureRelation:
    ...
```

`StructureRelation` 增加只读字段：

```python
symprec: float
cell_residual: float
position_residual: float
```

实施要求：

1. 入口验证 `symprec` 有限且严格为正；
2. 复制并 wrap 两个结构，保留 reference calculator；
3. 按第 3.2 节产生候选整数矩阵；
4. 完成纯整数的行列式、原子数和组成验证；
5. 按第 3.3 节计算并保存 `cell_residual`；
6. 超限异常必须包含实测残差、`symprec`、单位 Å 和候选矩阵；
7. 保留按元素的匈牙利指派和 `PeriodicGeometry.mic`；
8. 按第 3.4 节计算并保存 `position_residual`；
9. 原子映射超限异常必须包含失败原子、实测最大残差、`symprec` 和单位 Å；
10. 构造 `PeriodicIndex` 完成全局离散校验；
11. 可以继续把已验证标签写入 `reference.arrays`/`reference.info`，但这些元数据不能绕过当前几何验证；
12. `displacement` 使用保存的 `self.symprec`，按第 3.5 节检查胞残差。

不要把外部超胞要求为逐位等于工具生成结果。允许用户从文件、phonopy、pymatgen 或其他程序提供超胞，只要它在统一 `symprec` 下满足晶格和原子映射关系。

### P4：迁移全部核心调用方

逐项迁移：

| 调用点 | 要求 |
|---|---|
| `interactions/space.py` | `StructureRelation.from_atoms(..., symprec=symprec)`；删除 `tolerance=symprec` |
| `phonon/sscha/solver.py` | 同上；不得自行再检查一次映射 |
| `force_constants/realization.py` | 从源 relation 或调用上下文取得 `symprec` 并显式传入；不得恢复默认隐藏值 |
| `io/hdf5.py` | 对 $1 \times 1 \times 1$ identity relation 使用文件中记录的 `symprec`；若格式尚未保存它，应在 schema 中保存，或走不需要浮点映射的专用 identity 构造器 |
| `finite_difference/**` | 只通过共享 `ReferenceFrame` 使用 relation，不重复判定 |
| `fitting/**` | 同上 |

HDF5 的建议方案是为纯 identity relation 增加一个小而明确的离散构造器，例如 `StructureRelation.identity(primitive, *, symprec)`；它仍保存 `symprec` 供固定胞训练帧使用，但不需要从浮点反推矩阵和原子对应。不要为了 HDF5 给 `from_atoms` 恢复默认值。

所有顶层计算对象应只有一个 `symprec` 值，并把同一数值传给：

- `StructureRelation`；
- `PrimitiveSymmetryOperations`；
- `LatticeFrame`；
- interaction/orbit 构造。

禁止某个下游组件乘以常数后把它重新解释成结构身份阈值。现有 `symmetry.py` 中的 `symprec * 10.0` 必须单独审查：若它只是从 spglib 返回操作中恢复站点映射，应改为在 `symprec` 内匹配，或给出为什么需要放大的可复现反例；不得原样视为符合本计划。

### P5：处理非核心 `align_structures`

推荐把 `align_structures` 从 `structure/relation.py` 移到 `mlfcs.tools.structure_alignment`，原因是它会主动重排独立外部结构，而核心计算明确不允许静默重排训练帧。

目标接口：

```python
def align_structures(
    reference: Atoms,
    atoms: Atoms,
    *,
    tolerance: float,
) -> tuple[Atoms, float]:
    ...
```

要求调用方显式给 `tolerance`，不提供默认值。它可以分别计算胞残差和原子残差，但二者都使用这一个外部导入策略。生产包不得调用它。

如果本轮决定不移动它，则至少从 `mlfcs.structure` 公共导出中删除，并保持零生产调用者；不得让其 `tolerance` 混入核心阈值计数。

### P6：测试矩阵

#### P6.1 晶格关系

覆盖：

- 对角、非对角、负非零行列式整数矩阵；
- 立方、六方、倾斜原胞；
- adjacent、crossed、large shear 幺模换基；
- 大重复矩阵，至少包含系数 31；
- 晶格残差分别为 $0$、$0.5\,\mathrm{symprec}$、刚低于 `symprec`、等于 `symprec`、刚高于 `symprec`；
- 无量纲矩阵差很小但 Å 残差超限的拒绝例；
- 无量纲矩阵差较大但归一化 Å 残差合法的接受例。

断言候选整数矩阵、`cell_residual` 和错误消息中的数值。

#### P6.2 原子映射

覆盖：

- reference 任意原子重排；
- primitive 任意站点重排；
- 同元素近邻需要全局指派、贪心会失败的案例；
- 不同元素不能互换；
- 原子数、化学计量、陪集重复和陪集缺失；
- 原子 MIC 残差在阈值下、边界上和阈值上；
- 非正交胞中普通分数坐标 wrap 不是最短像的案例。

现有“需要 `tolerance=0.6` 才接受 H2 模糊映射”的测试改为 `symprec=0.6`。它测试的是全局指派算法，不是为第二个容差提供理由。

#### P6.3 训练帧

覆盖：

- 相同胞、正常原子位移被接受；
- 元素顺序改变被拒绝；
- 胞残差在边界下、边界上和边界上方；
- 原子真实位移可以远大于 `symprec`，仍正确返回，不被误判为映射失败。

#### P6.4 架构

新增或扩展 `tests/test_architecture_dependencies.py`：

1. `tools` 允许依赖 `structure`；
2. 所有主线包都禁止依赖 `tools`；
3. `structure/supercell.py` 不再存在；
4. `mlfcs` 和 `mlfcs.structure` 不导出 `build_supercell`；
5. 所有需要超胞的公共计算入口的 `reference` 都是无默认值的必需关键字参数。

#### P6.5 属性与物理回归

对现有 cubic、fcc primitive、hcp 和倾斜晶胞案例，验证：

- 相同 `symprec` 下，重构前后原胞—超胞标签相同；
- 轨道数、参数数、观测行和有限差分计划指纹符合预期；
- IFC 和教程指标不发生超出既有数值误差的改变；
- 若 plan fingerprint 因 API/schema 有意改变，旧 `ForceBatch` 必须被明确拒绝，不得兼容读取。

## 6. 文档和教程迁移

至少更新以下中英文页面：

- `reference/structures-api.md`
- `reference/finite-difference-api.md`
- `reference/fitting-api.md`
- `reference/sscha-api.md`
- `reference/units-and-parameters.md`
- `theory/symmetry-and-orbits.md`
- `theory/finite-supercells.md`

文档必须说明：

1. `reference` 是用户提供的物理超胞，不由 cutoff 或矩阵隐式生成；
2. `symprec` 是唯一的核心长度精度，单位为 Å；
3. `angle_tolerance` 留给 spglib 自动处理；
4. `build_supercell` 是 `mlfcs.tools` 下的可选便利函数，主线不依赖它；
5. MLFCS 接受任何在 `symprec` 内与原胞构成整数复制关系的显式超胞，不要求来自本工具。

更新 `CHANGELOG.md` 和 `CHANGELOG_ZH.md`，明确列出删除的导入路径和 `tolerance` 到 `symprec` 的破坏性改名。不得写“deprecated”；这是直接删除。

所有 Markdown 数学继续只用 `$...$` 和 `$$...$$`，不得引入其他数学定界符。

教程中每个独立拟合任务仍必须由任务脚本自行覆盖写入本目录固定 `fit.log`，包含完整 stdout、stderr 和 traceback；本重构不得引入公共日志包装器。

## 7. 提交拆分

建议按以下可独立审查的顺序提交：

1. `tests: lock the explicit-supercell and single-symprec contract`
2. `tools: isolate optional supercell construction from the core DAG`
3. `structure: make integer-lattice inputs strictly discrete`
4. `structure: validate primitive-supercell relations with symprec`
5. `api: migrate relation consumers to the single symmetry precision`
6. `tools: isolate explicit external-structure alignment`
7. `docs: document the explicit-supercell 4.0 break`
8. `tutorials: regenerate affected plans, logs, metrics, and plots`

每个提交只包含对应阶段，聚焦测试在该提交上必须通过。不要把教程二进制图、核心实现和文档混在同一提交。

## 8. 验收命令

实现者应在交付报告中给出下列命令的完整结果：

```bash
rg -n "mapping_tolerance|cell_tolerance|position_tolerance" src tests
rg -n "tolerance=symprec|atol=1e-7|atol=1e-10" src tests
rg -n "from mlfcs(\.structure)? import build_supercell" .
rg -n "mlfcs\.tools" src/mlfcs \
  -g '!tools/**'
pytest -q tests/test_core_structure_relation.py
pytest -q tests/test_tools_supercell.py
pytest -q tests/test_architecture_dependencies.py
pytest -q tests/test_public_api_contract.py
ruff check src tests
python scripts/check_docs.py
mkdocs build --strict
python scripts/sync_readmes.py --check
pytest -q
```

前两条搜索应无生产代码匹配；第三条只允许迁移说明或负面测试；第四条必须无输出。

全量测试如果存在基线既有失败，必须提供基线提交上的同命令结果，不能只口头声明“既有”。

## 9. 交付报告模板

实现者最终必须报告：

1. 基线提交、候选 HEAD 和逐提交列表；
2. 删除的公共 API 与新的唯一导入路径；
3. 所有要求显式 `reference` 的入口清单；
4. `symprec` 在核心中 3 个比较位置及 spglib 转交位置；
5. 最大 `cell_residual`、最大 `position_residual` 及对应测试结构；
6. 阈值边界测试结果；
7. 主线不依赖 `tools` 的 AST/DAG 证据；
8. 全套测试、ruff、文档门禁结果；
9. 重跑的教程任务及每个 `fit.log` 路径；
10. 所有数值产物差异；若不为逐位相同，说明物理原因和误差量级；
11. 未跟踪文件和未纳入提交的既有修改。

## 10. 明确不在本计划内

- 不重写 spglib 的对称性搜索；
- 不暴露 `angle_tolerance`；
- 不做 primitive cell 自动识别；用户提供的 `primitive` 就是模型原胞；
- 不从 cutoff 推导超胞；
- 不把原胞—超胞映射改成精确分数坐标相等；
- 不删除匈牙利全局指派；
- 不允许变胞训练；
- 不修改倒空间约化、SCPH 算法或 SSCHA 采样物理；
- 不处理 ASR、拟合求解器或 IO 文本解析中的其他容差；
- 不顺带修复与本计划无关的 DAG 缺陷或格式漂移。

## 11. 完成定义

只有同时满足以下条件才可合并：

1. 用户必须显式提供参考超胞；
2. `build_supercell` 只存在于 `mlfcs.tools.supercell`；
3. 主线包对 `mlfcs.tools` 零依赖；
4. 核心原胞—超胞关系只暴露 `symprec`，不存在 `mapping_tolerance` 等第二精度；
5. 无量纲量不与 Å 制的 `symprec` 直接比较；
6. `StructureRelation` 的三处几何判定全部使用同一个保存的 `symprec`；
7. `normalize_supercell_matrix` 不再用浮点阈值猜整数；
8. 外部轻微扰动超胞在 `symprec` 内可被接受，超限时错误报告包含实际 Å 残差；
9. 架构、聚焦、全量、文档和教程门禁均有可复核结果；
10. 没有兼容别名、静默 fallback 或隐藏容差。
