# MLFCS `structure` 模块分析与重构方案

状态：Phase 1–8 分析稿，尚未实施代码重构。

本文基于当前 `src/mlfcs/structure/`、直接消费者、测试、教程和现有中英文文档编写。它是重构审查材料，不是对最终 API 的承诺。文中所有坐标、索引和数组 shape 均按当前实现解释。

## 1. 结论摘要

`structure` 当前不是一个单一的结构数据容器，而是四类紧密相关的基础能力集合：

1. ASE `Atoms` 的 primitive/reference 关系验证和位移提取；
2. 周期几何和最小像计算；
3. 整数晶格商空间、HNF 归约和 reciprocal quotient；
4. primitive/supercell 原子索引以及 spglib 对称操作的具体化。

这个边界总体合理，因为这些能力共同维护 MLFCS 的核心约定：原子顺序、primitive site、整数 lattice translation 和 reference supercell 的一一对应关系。当前最主要的问题不是功能缺失，而是边界和所有权表达不够明确：

- `StructureRelation` 复制并修改 ASE `Atoms`，同时把映射写入 `reference.arrays` 和 `reference.info`；
- `PeriodicIndex`、`IntegerLatticeQuotient` 和 `StructureRelation` 分别保存相互关联的 mapping/quotient 派生状态；
- `symmetry.py` 同时保存 primitive affine 操作和 reference atom permutation，两个不同数据空间没有在类型名中区分；
- `structure.reciprocal` 已经被 `phonon.reciprocal` 转发，容易把通用 lattice quotient 误认为声子专属逻辑；
- 若干下游模块直接导入内部文件，而不是从 `mlfcs.structure` 使用稳定出口；
- `PeriodicGeometry` 的 closest-image cache 是可变且没有容量策略的内部状态，冻结 dataclass 并不意味着对象完全不可变；
- `IntegerLatticeQuotient` 对三维整数矩阵的通用处理合理，但 `reciprocal.py` 中 reciprocal 物理语义和整数商空间数学语义混在同一层。

建议采用一次小步、保留兼容的重构：先把数据空间和 ownership 写清楚，再只做低风险模块拆分和出口整理。不建议新建 `StructureFactory`、`StructureManager` 或通用 `StructureContext`，也不建议在本次任务中把 ASE `Atoms` 替换成新的大数据模型。

## 2. 当前调用和数据流

### 2.1 创建者和消费者

结构对象的主要创建入口是：

```text
ASE Atoms / primitive + reference
    -> StructureRelation.from_atoms
        -> normalize_supercell_matrix
        -> PeriodicGeometry.mic
        -> linear_sum_assignment
        -> PeriodicIndex

ASE primitive
    -> PrimitiveSymmetryOperations.from_atoms
        -> spglib.get_symmetry_dataset

primitive + integer supercell matrix
    -> build_supercell
        -> phonopy cell builder (or local fallback)
```

主要消费者如下：

```text
StructureRelation
    -> fitting.dataset / fitting
    -> finite_difference
    -> force_constants.materialize / realization
    -> calculators.taylor
    -> io.hdf5 / io.alamode / io.shengbte / io.phonon_hdf5
    -> phonon.scph / phonon.sscha

PeriodicIndex
    -> interactions.realization
    -> finite_difference.reconstruction
    -> force_constants.dense / io writers
    -> phonon.sampling.harmonic

PrimitiveSymmetryOperations
    -> interactions.primitive.builder
    -> SymmetryOperations.from_primitive_operations
    -> symmetry tests

SymmetryOperations
    -> interactions.space
    -> core symmetry tests

PeriodicGeometry
    -> StructureRelation
    -> alignment
    -> cluster candidate distance helpers
    -> IO export geometry

IntegerLatticeQuotient / reciprocal quotient
    -> structure mapping
    -> phonon reciprocal and SCPH q-point construction
    -> integer-lattice and anharmonic q-point tests
```

### 2.2 端到端数据流

```text
输入 ASE Atoms
    primitive: 周期 primitive cell，原子顺序是 primitive site 顺序
    reference: 周期 reference/supercell，原子顺序是外部 reference 顺序
        |
        v
验证 cell(reference) = S * cell(primitive)
并将 S 归一化为 int64 3x3 矩阵
        |
        v
按化学元素和 HNF 商空间生成候选 primitive-site/cell slots
        |
        v
PeriodicGeometry + Hungarian assignment 建立 reference atom ->
primitive site + integer translation
        |
        v
StructureRelation
    primitive/reference copies
    supercell_matrix
    primitive_index[n_reference]
    cell_translation[n_reference, 3]
    PeriodicIndex 派生查找表
        |
        +--> interaction realization / fitting / finite difference / IFC IO
        +--> primitive symmetry -> reference symmetry
        +--> exact-R force-constant materialization
```

## 3. 数据空间和 source of truth

| 数据 | 当前表示 | 当前拥有者 | 主要消费者 | 结论 |
| --- | --- | --- | --- | --- |
| primitive cell | ASE `Atoms` | `StructureRelation.primitive` 或调用者 | symmetry、interaction、fitting | primitive `Atoms` 是结构 source of truth |
| reference cell | ASE `Atoms` | `StructureRelation.reference` 或调用者 | fitting、FD、IO、materialization | reference `Atoms` 是具体有限视图 source of truth |
| Cartesian positions | `Atoms.positions`, shape $(n,3)$ | ASE `Atoms` | MIC、匹配和外部写出 | 不在 structure 中复制 |
| fractional positions | `Atoms.get_scaled_positions()` | ASE `Atoms` 派生值 | spglib、supercell builder | 使用时显式转换，不单独拥有 |
| atomic numbers | `Atoms.numbers`, shape $(n,)$ | ASE `Atoms` | 化学种类匹配、spglib | 不应由 mapping 再保存一份 |
| pbc/cell | ASE `Atoms` | ASE `Atoms` | geometry、relation 验证 | relation 只验证和引用 copy |
| supercell matrix | `int64 (3,3)` | `StructureRelation` / `PeriodicIndex` | quotient、mapping、IO | 应保证一个规范化 source |
| primitive index | `int32 (n_reference,)` | `StructureRelation`，也写入 reference arrays | `PeriodicIndex`、materialization | relation 字段是逻辑 source of truth；Atoms array 是兼容 metadata |
| cell translation | `int32 (n_reference,3)` | `StructureRelation`，也写入 reference arrays | exact-R mapping、IO | relation 字段是逻辑 source of truth |
| quotient/HNF | `IntegerLatticeQuotient` 派生缓存 | `PeriodicIndex` | atom lookup、reciprocal | 不应由调用者修改 |
| closest-image results | geometry cache | `PeriodicGeometry` | cluster/image logic | 纯缓存，不是物理 source of truth |
| primitive symmetry | spglib-derived arrays | `PrimitiveSymmetryOperations` | orbit builder、reference symmetry | spglib 是 affine operation source |
| reference atom permutation | derived array | `SymmetryOperations` | finite/reference symmetry consumers | 仅是给定 relation 下的视图 |

注意：当前 `frozen=True` 只冻结 Python 属性重新绑定；ASE `Atoms`、NumPy arrays 和 geometry dict 仍可变。后续若要强化不可变语义，必须先确认消费者没有依赖这些对象的原地修改，不能只加 `writeable=False`。

## 4. 实体级 API 分析

### 4.1 `integer_lattice.py`

| API | 输入和输出 | 职责与状态 |
| --- | --- | --- |
| `normalize_supercell_matrix(matrix)` | `(3,)` repeats 或 `(3,3)` 数值；返回 `int64 (3,3)` | 验证整数性、非奇异性和 int64 边界；无副作用 |
| `determinant_3x3(matrix)` | `int64 (3,3)`；返回 Python `int` | 精确三维行列式；无副作用 |
| `adjugate_3x3(matrix)` | `int64 (3,3)`；返回 `int64 (3,3)` | 精确伴随矩阵；无副作用 |
| `residue_key(translation,matrix)` | 整数平移 `(3,)`；返回 3 元组 | 计算商空间 residue；无状态 |
| `same_residue(a,b,matrix)` | 两个整数平移；返回 bool | residue 等价判断 |
| `row_hermite_normal_form(matrix)` | 非奇异整数矩阵；返回 lower-triangular `int64 (3,3)` | 复用 SymPy HNF，固定 MLFCS row-lattice convention |
| `IntegerLatticeQuotient` | `matrix`；派生 `hnf`、representatives、strides | 维护 $Z^3 / Z^3 S$ 的规范代表和索引 |
| `.decompose/decompose_many` | 整数平移；返回 quotient/remainder | 精确 HNF 分解，不修改输入 |
| `.reduce/reduce_many` | 整数平移；返回 residue | 规范化平移 |
| `.cell_index/cell_index_many` | 平移；返回整数 cell id | mixed-radix deterministic index |
| `.equivalent` | 两个平移；返回 bool | 商空间等价判断 |

当前设计的优点是三维整数 arithmetic 和浮点几何明确分开；主要风险是 `matrix` 同时在 quotient、index 和 relation 中保存，存在派生状态重复。

### 4.2 `periodic_geometry.py`

`PeriodicGeometry` 接收 nonsingular Cartesian cell `(3,3)`、全周期 `pbc` 和 tolerance。它使用 ASE `find_mic` 得到一般最小像，使用 ASE `minkowski_reduce` 产生局部搜索基，再通过有限局部候选恢复并列最近像。

| API | 输入和输出 |
| --- | --- |
| `mic(vectors)` | Cartesian `(...,3)` 实际传入可广播的二维数组；返回 MIC vectors 和 lengths |
| `pair_distances(positions)` | Cartesian `(n,3)`；返回 `(n,n)` 距离矩阵 |
| `closest_images(vector)` | 一个 Cartesian `(3,)`；返回 image vectors 和 integer shifts `(k,3)` |
| `minimum_length(vector)` | Cartesian `(3,)`；返回 float |
| `joint_closest_image_shifts(vectors)` | anchor-to-tail Cartesian `(n,3)`；返回 `(k,n,3)` shifts |
| `unique_periodic_distances(values)` | 距离数组；返回升序非零 list |

它不负责 primitive mapping、cluster identity、force constants 或 symmetry。`joint_closest_image_shifts` 虽被 cluster/IO 使用，但仍只解决几何兼容性，领域组合逻辑在消费者中。

### 4.3 `relation.py`

`StructureRelation` 是当前最重要的结构域对象。它表示一个 primitive 与一个具体 reference 的已验证关系，而不是一般意义的“两个结构相似”。

输入约定：两个周期 ASE `Atoms`；reference cell 必须满足 `reference.cell = S @ primitive.cell`，其中 $S$ 是 full-rank integer `(3,3)`；原子数必须等于 `abs(det(S)) * len(primitive)`。

`from_atoms` 的实际过程：复制输入、wrap 两者、验证 cell 和 atom count；为每种元素生成 primitive-site/coset slots；用 reference-cell MIC 距离矩阵和 `linear_sum_assignment` 做一一匹配；构造并验证 `PeriodicIndex`；将 `primitive_index`、`cell_translation`、`primitive_scaled_position` 和 `mlfcs_supercell_matrix` 写入复制后的 reference。

公开输出：

- `.primitive` 和 `.reference`：复制后的 ASE `Atoms`；
- `.supercell_matrix`：规范化 `int64 (3,3)`；
- `.primitive_index`：reference atom 到 primitive site 的 `int32 (n_reference,)`；
- `.cell_translation`：对应 primitive lattice translation 的 `int32 (n_reference,3)`；
- `.position_residual`：匹配最大 MIC residual；
- `.index`：只读访问的 `PeriodicIndex`；
- `.displacement(atoms)`：保持 reference 原子顺序，返回 Cartesian MIC displacement `(n_reference,3)`，不重排输入。

`align_structures(reference, atoms)` 是独立的显式重排工具：返回按 reference 顺序排列的 ASE `Atoms` 和最大匹配 residual。它不创建 `StructureRelation`，也不应被 fitting 隐式调用。

当前隐式行为包括：输入被 copy/wrap；reference copy 会获得 MLFCS metadata；`reference.calc` 从原 reference 复制；匹配使用 tolerance 和 reference-cell MIC。这些行为需要在 docstring 和测试契约中明确保留。

### 4.4 `supercell_mapping.py`

`PeriodicIndex` 表示一个固定 reference ordering 下的 O(1) 查找器。它接收：

- `primitive`: `(n_reference,)` 的 contiguous primitive site labels；
- `translations`: `(n_reference,3)` 整数 lattice translations；
- `supercell_matrix`: 可规范化整数矩阵。

构造时建立 `IntegerLatticeQuotient`、`atom_by_site_cell[n_primitive,n_cells]` 和 `translation_by_cell[n_cells,3]`。要求每个 primitive site 的每个 quotient coset 恰好出现一次。

| API | 输入和输出 |
| --- | --- |
| `.n_primitive` / `.n_cells` | scalar metadata |
| `.cell_representatives` | copy of canonical HNF translations `(n_cells,3)` |
| `.residue(translation)` | 3-tuple residue |
| `.canonical_translation(translation)` | reference ordering 中的 translation copy `(3,)` |
| `.atom(primitive,translation)` | 一个 supercell atom index |
| `.atom_many(primitive,translations)` | NumPy broadcasting 后的 atom indices |
| `.translate_atom(atom,shift)` | 一个 translated atom index |
| `.translate_atoms(atoms,shifts)` | batched atom indices |
| `.anchor(cluster)` | 把 cluster 第一 atom 平移到零 translation 后的 tuple |
| `.representative(primitive)` | zero-translation reference atom index |

它不拥有 Cartesian positions，也不验证 atom geometry；它只消费已验证的 discrete relation。这个边界应保留。

### 4.5 `symmetry.py`

`PrimitiveSymmetryOperations` 保存 spglib 给出的 primitive affine operation：

- rotations `(n_ops,3,3)` integer scaled-coordinate rotations；
- translations `(n_ops,3)` float fractional translations；
- cartesian rotations `(n_ops,3,3)`；
- site permutations `(n_ops,n_primitive)`；
- site shifts `(n_ops,n_primitive,3)` integer translations；
- international symbol。

`from_atoms` 是唯一的晶体对称识别入口，负责调用 spglib 并把每个 transformed primitive site 解析成 site permutation 和整数 shift。`transform_label(operation,(site,tx,ty,tz))` 作用于 exact-R site label，返回相同 label 空间。

`SymmetryOperations` 保存 primitive operations 在一个具体 reference 上的实现：筛选与 supercell quotient 兼容的 rotations，继承 affine arrays，并生成 `atom_permutations[n_ops,n_reference]`。它是 relation-dependent finite reference view，不是新的空间群识别器。

目前两个类的 operation array 命名过于相似，建议通过 docstring 和类型别名/字段注释明确 `scaled primitive operation`、`exact-R site operation`、`finite reference atom permutation` 三个层次；不建议立即重命名字段，因为这会扩大兼容风险。

### 4.6 `supercell.py`

`build_supercell(primitive, supercell_matrix, symprec)` 是结构生成入口。它只接受周期 ASE primitive 和整数矩阵，优先调用 phonopy `get_supercell(..., is_old_style=True)` 产生 MLFCS/phonopy 需要的 old-style ordering；phonopy 不可用时走本地 fallback。

输出是新的 ASE `Atoms`，包含 cell、wrapped scaled positions、symbols/numbers 和 `pbc=True`。它目前不显式写入 `primitive_index`、`cell_translation` 或 `mlfcs_supercell_matrix`；这些 metadata 由 `StructureRelation.from_atoms` 在 relation 建立时补齐。此分工应保持，否则会产生两套 mapping source。

### 4.7 `reciprocal.py`

`ReciprocalQuotientGrid` 和 `reciprocal_quotient_grid` 把整数 supercell matrix 转为有限 reciprocal quotient：先对 `matrix.T` 建立 HNF representatives，再通过 adjugate 和 determinant 构造 exact labels，返回 labels、denominator 和 fractional points。`negative_label` 以 denominator 为模取负。

数学上它属于整数晶格/有限商空间；当前被 `mlfcs.phonon.reciprocal` 重新导出给声子模块。建议保留通用实现于 `structure`，让 `phonon.reciprocal` 只负责声子语义的 façade 或明确的 q-point命名，避免通用 lattice code 反向依赖 phonon。

## 5. 为什么需要独立 `structure`

直接让所有模块操作 ASE `Atoms` 会导致以下约定在各处重复实现：

- fractional 和 Cartesian 坐标转换；
- reference atom 与 primitive site 的对应；
- translation 的整数表示和 quotient reduction；
- skewed/non-diagonal supercell 的 cell mapping；
- MIC 和 tie image；
- spglib affine operation 到 exact-R/reference permutation 的转换。

这些约定一旦在 fitting、IO、finite difference 和 force-constant materialization 中分别实现，就会出现同一个原子在不同模块中拥有不同 index 或 translation。`structure` 的实际价值不是取代 ASE，而是验证和集中维护这些离散/几何约定。ASE `Atoms` 仍应是外部结构数据的 source of truth；MLFCS 对象只拥有已验证的关系和派生 lookup。

## 6. 职责边界

### 应属于 `structure`

- 周期 ASE 结构的基本验证和显式复制/wrap 规则；
- primitive/reference 的整数超胞关系；
- atom mapping 和 translation mapping；
- 周期 MIC、nearest-image 和 joint image 几何 primitive；
- integer lattice normalization、determinant、HNF quotient 和 exact reciprocal quotient；
- primitive affine symmetry 的 spglib 适配；
- 已知 relation 下的 reference atom permutation；
- 按既定 ordering 生成 supercell。

### 不应属于 `structure`

- cluster candidate/orbit enumeration；
- InteractionKey、Taylor basis、stabilizer Gram 和 invariant basis；
- fitting/design/Gram accumulation；
- force-constant sparse/dense materialization；
- ASR、rotational sum rules 和 reconstruction；
- q-point phonon modes、dynamical matrix、SCPH、SSCHA；
- IO 文件格式的业务规则（但 IO 可以消费 structure 的 mapping）；
- force calculation、Taylor evaluation 和 calculator 行为。

当前代码总体遵守上述边界。`structure.reciprocal` 是最接近边界的文件：其实现是 lattice quotient，不是 phonon solver，因此建议从命名和文档上澄清，而不是把它搬进声子算法层。

## 7. 当前问题审计

| 问题 | 影响 | 原因 | 建议 |
| --- | --- | --- | --- |
| `StructureRelation` 是 frozen，但内部 `Atoms`/arrays 可变 | 调用者可能让 relation 派生 mapping 与结构不一致 | dataclass freeze 只限制属性绑定 | 本次先文档化；后续再评估 defensive copy 或只读 view |
| mapping 同时存在 relation 字段、reference arrays 和 `PeriodicIndex` lookup | 维护和序列化时容易误判 source | 为 IO/兼容保留 metadata，但 ownership 未写明 | relation 字段作为 source；Atoms arrays 作为 compatibility metadata；index 作为 cache |
| `supercell_matrix` 在 relation 和 index 中各保存一份 | 修改/比较语义不清 | index 需要独立构造验证 | 保持值复制，禁止原地修改；后续考虑明确 `relation.index` 为唯一 lookup owner |
| `symmetry.py` 中 affine、site、finite atom permutation 混在两个相近命名类中 | 读者不易判断 operation 所属数据空间 | 当前字段名偏通用 | 先补 docstring/type comments，不做无必要 rename |
| `PeriodicGeometry` 的 cache 无容量上限 | 大量不同 displacement 或 cluster 时内存可增长 | dataclass 内部 dict 直接缓存 | 先测量实际命中率；后续可增加可选 bounded cache，但不在分析阶段改行为 |
| `PeriodicGeometry._reduction` 通过 `rint` 转 int32 | 极端 cell/数值下可能掩盖 reduction 返回的整数约定 | ASE 返回的 reduction transform 未在类型上表达 | 加 invariant test 和注释，确认版本契约后再改 |
| `reciprocal.py` 通过 `phonon.reciprocal` 重导出 | 通用数学能力的所有权不直观 | phonon API 需要 q-point 入口 | 保留 structure 实现，phonon 只做明确 facade |
| 许多消费者导入内部文件路径 | 后续文件移动会扩大迁移面 | 历史 API 形成 | `structure.__init__` 增加稳定公开出口前先盘点冲突，再逐步迁移 |
| `__init__.py` 未导出 symmetry 和 integer lattice public helpers | 用户和测试只能使用内部模块路径 | 早期 API 只暴露常用结构入口 | 是否扩大公开 API 需审查；不要自动把所有 helper 顶层化 |
| `build_supercell` 的 phonopy fallback 与主路径可能有 ordering 差异 | 无 phonopy 环境下下游 mapping 可能不同 | fallback 只模拟 old-style ordering | 保留独立回归，明确 fallback 是兼容实现 |
| `align_structures` 用 Hungarian assignment，但不是 relation builder | 名称相近时可能被误作 relation 建立入口 | 两者均做结构匹配 | docstring 明确 align 是显式重排 utility，relation 是 verified mapping |

未发现必须通过大规模抽象才能解决的循环 import。当前 import 方向是：integer lattice -> geometry/mapping/relation -> higher-level consumers；`structure` 没有反向依赖 fitting、interaction 或 phonon solver，属于健康方向。

## 8. 推荐目标架构

推荐只做职责显式化，不引入新的通用 manager/builder/context 类：

```text
mlfcs.structure
├── __init__.py                 稳定结构基础出口
├── lattice.py                  可选：整数 lattice API 的 façade；不复制实现
├── integer_lattice.py          int64 matrix、HNF、quotient
├── geometry.py                可选：periodic geometry 的新语义名
├── periodic_geometry.py        兼容实现，后续可保留为实现模块
├── relation.py                 primitive/reference verified relation
├── mapping.py                  可选：PeriodicIndex 的 façade
├── supercell.py                supercell generation
├── supercell_mapping.py        PeriodicIndex 实现
├── symmetry.py                 primitive/reference symmetry representations
└── reciprocal.py               generic reciprocal quotient
```

但是，第一阶段不建议立即执行上述可选重命名。低风险目标实际上是：

```text
现有文件实现保持
    -> 补充清晰 docstring 和数据空间说明
    -> 明确 __init__ 稳定出口
    -> 将 reciprocal 的通用性质写清楚
    -> 逐步把消费者从内部路径迁移到稳定出口
    -> 测试锁定 ordering、mapping、HNF、MIC 和 symmetry 数值行为
```

如果审查后确认需要物理语义重命名，再考虑 `geometry.py`/`mapping.py` façade；不要同时移动实现、改名 API 和改数据模型。

## 9. 文件级实施计划

### 保留并优先补充文档

- `src/mlfcs/structure/integer_lattice.py`：保留实现；补充 row-vector、HNF remainder 和 overflow 约定。
- `src/mlfcs/structure/periodic_geometry.py`：保留实现；补充输入 shape、Cartesian 约定和 cache 语义。
- `src/mlfcs/structure/relation.py`：保留实现；补充 ownership、copy/wrap 和 mapping source of truth。
- `src/mlfcs/structure/supercell.py`：保留实现；补充 phonopy old-style ordering 和 fallback 限制。
- `src/mlfcs/structure/supercell_mapping.py`：保留实现；补充 primitive-site/coset invariant。
- `src/mlfcs/structure/symmetry.py`：保留实现；补充每个 operation array 的数据空间。
- `src/mlfcs/structure/reciprocal.py`：保留实现；明确它是 generic lattice quotient，不拥有 phonon modes。

### 小范围修改

- `src/mlfcs/structure/__init__.py`：审查并稳定化出口；是否公开 lattice/symmetry 类型需根据兼容矩阵决定。
- `src/mlfcs/phonon/reciprocal.py`：保留 facade，但在 docstring 中说明实现 source 在 `structure.reciprocal`。
- `src/mlfcs/io/*`、`src/mlfcs/fitting/*`、`src/mlfcs/finite_difference/*`、`src/mlfcs/interactions/*`：只把可安全的内部 import 逐步切换到稳定出口，不改对象或算法。

### 测试整理/新增

- 保留 `test_integer_lattice.py`、`test_core_structure_relation.py`、`test_core_supercell.py`、`test_core_supercell_builder.py`、`test_core_symmetry.py`、`test_core_neighbors.py`。
- 新增或补强 relation source-of-truth、input copy/wrap、reference metadata、non-diagonal matrix、negative determinant rejection、MIC tie ordering 和 phonopy/fallback ordering 的测试。
- 在 architecture tests 中锁定 `structure` 不反向依赖 fitting/interaction/phonon solver；允许 phonon 消费 generic reciprocal/structure mapping。
- 不删除现有数学 oracle 或 IO 测试；重构前后以代表结构的 cell、positions、numbers、mapping、translation 和 output ordering 做快照对照。

### 暂不做

- 不删除 `structure` 旧文件；
- 不把 ASE `Atoms` 替换成新 `Structure` class；
- 不把 `PeriodicIndex` 合并进 `StructureRelation` 的公开数据字段；
- 不把 reciprocal quotient 搬入 phonon；
- 不修改 cluster/orbit/fitting/force-constant 算法；
- 不改变 primitive/reference atom ordering、HNF convention 或 numerical tolerances；
- 不删除旧 import/API，除非后续审查明确批准兼容策略。

## 10. 兼容性风险

| 风险 | 具体表现 | 控制方法 |
| --- | --- | --- |
| import compatibility | 下游直接使用 `mlfcs.structure.relation` 等内部路径 | 先保留路径，新增出口后逐步迁移 |
| atom ordering | phonopy old-style 与 fallback 顺序变化会改变 IFC 文件 | golden test 对比 atom mapping 和写出结果 |
| mapping identity | relation copy 与原输入不是同一对象 | 保持当前 copy 语义并写明，不把 identity 当契约 |
| coordinate convention | positions 是 Cartesian，scaled positions 是 fractional | 所有 docstring 明确，测试同时检查两种坐标 |
| translation convention | row-vector integer translation、HNF lower triangular、模 quotient | 保留 exact integer tests，不使用浮点等价替换 |
| symmetry ordering | spglib operation order 和 selected reference operations | 不排序、不重算 group；快照比较 operation arrays |
| numerical behavior | MIC/Hungarian/tolerance 变化会影响 relation residual | 不改算法和 tolerance，先补测试 |
| mutability | frozen dataclass 内部对象仍可写 | 先不改变行为，后续单独设计 immutable view |
| external metadata | IO 读取依赖 `primitive_index`、`cell_translation`、matrix info | relation 建立时继续写入 metadata，写入格式测试 |

## 11. 审查结论与下一步

Phase 1–8 的建议是：批准后先做“文档和测试契约 + 稳定出口”这一小步，再评估是否需要真实文件拆分。当前没有证据表明拆分成大量新 class 会改善数值代码；真正需要优先解决的是数据空间、source of truth 和 ordering 约定的可见性。

本稿完成后按任务纪律停在审查点。获得明确批准后，实施顺序应为：

1. 补充结构 API docstring 和关键数学/ordering 注释；
2. 加强 mapping、copy、fallback 和数据空间测试；
3. 稳定 `mlfcs.structure` 出口并迁移安全的内部 imports；
4. 运行 structure 相关测试及全量回归；
5. 只有在回归证明没有行为差异后，才考虑可选 façade 或文件拆分；
6. 最终开发文档继续写在 `/home/gwins/codespace/mlfcs-new/architecture/`。
