# 周期平移 residue 规范化计划

## 目标

为一般整数 3x3 超胞矩阵建立严格、独立于 phonopy 的周期平移 residue 实现，用于 primitive site 与 reference atom 查找、source/target materialization、alias collision 检测、对称操作索引归一化和有限超胞平移代表枚举。

本计划只处理整数晶格商群，不处理真实空间最近镜像、cutoff、Wigner-Seitz 或格式专用 27 镜像。

## 数学定义

对行向量约定的整数超胞矩阵 `S`，两个平移 `R1` 和 `R2` 属于同一个 residue，当且仅当：

```text
R1 - R2 = n @ S,    n in Z^3
```

residue 是商群 `Z^3 / Z^3 S`，其元素数量严格为 `abs(det(S))`。

## 当前实现审计

当前 `src/mlfcs/core/geometry.py` 已经使用商群语义：

- `_translation_label()` 用伴随矩阵和行列式构造 residue label；
- `_coset_translations()` 用 BFS 枚举 residue 代表；
- `PeriodicIndex` 检查每个 primitive site 是否包含完整的 `abs(det(S))` 个 residue；
- 重复 `(primitive_site, residue)` 会被拒绝。

当前实现的问题是：

1. 伴随矩阵通过浮点 `det/inv/rint` 恢复，不是纯整数计算；
2. BFS 代表元依赖遍历顺序，代表坐标可能较大；
3. 代表元没有明确的有界混合进制规范；
4. residue 与真实空间最近镜像的职责需要在 API 和文档中进一步分离。

## 目标架构

### 1. 精确 residue 等价性

实现纯整数 3x3 运算：

- `determinant_3x3(S)`；
- `adjugate_3x3(S)`；
- `same_residue(R1, R2, S)`；
- `residue_key(R, S)`。

不使用浮点坐标、不使用 MIC、不使用 Wigner-Seitz。对 `Delta = R1 - R2`，使用 `adj(S) @ Delta` 和 `det(S)` 做精确整除判定。

### 2. HNF 代表元枚举

需要完整 residue 集合时，将超胞平移格转换为等价的 Hermite normal form `H`，使用三角整数矩阵的有界坐标枚举恰好 `abs(det(S))` 个代表。

HNF 只用于：

- `build_supercell` 工具；
- 新建 `PeriodicIndex` 时生成完整平移表；
- dense materialization；
- 需要完整 target translation lookup 的导出场景。

HNF 不用于真实空间最近镜像选择。

### 3. 用户直接提供 reference supercell

如果 reference supercell 由用户直接提供：

1. `StructureRelation` 通过元素分块和周期匹配得到每个 atom 的 primitive site 与整数平移；
2. 直接对这些平移执行 residue 验证；
3. 检查每个 primitive site 有恰好 `abs(det(S))` 个不同 residue；
4. 建立 `(primitive_site, residue) -> atom_index` 查找表。

这种路径不需要重新枚举 HNF 代表元。HNF 只作为独立验证或需要构造完整 target 表时调用。

### 4. 与 Wigner-Seitz 的职责分离

residue 层只回答：

```text
两个整数平移是否属于同一周期位置？
```

它不负责最近镜像、全部简并最近镜像、cutoff 邻居或 ShengBTE/ALAMODE 的格式平移。这些由 `PeriodicGeometry`、`Materializer` 和具体 writer 负责。

## 外部依赖策略

本实现不依赖 phonopy、SymPy 或 SageMath。

- phonopy 的 `SNF3x3` 是其自身实现，不是通用运行时库；
- SymPy 适合求 HNF/SNF 形式，但不保证提供我们需要的完整变换矩阵；
- SageMath 依赖过重，不适合作为 MLFCS 运行时依赖。

HNF/SNF 的数学定义用于验证和文档；运行时使用自有的轻量 3x3 整数实现。

## API 设计

内部 API 建议：

```python
same_residue(translation_a, translation_b, supercell_matrix) -> bool
residue_key(translation, supercell_matrix) -> tuple[int, int, int]
enumerate_residue_representatives(supercell_matrix) -> np.ndarray
```

`PeriodicIndex` 只暴露 `residue(translation)`、`atom(primitive_site, translation)` 和 `canonical_translation(translation)`。公共 API 不暴露 SNF/HNF 的内部变换矩阵。

## 验证标准

### 代数正确性

- 对角超胞与逐坐标模运算结果一致；
- 非对角超胞 residue 数量等于 `abs(det(S))`；
- 负平移与正平移的等价性正确；
- 任意整数 `n` 满足 `same_residue(R, R + n @ S, S)`；
- 不等价平移不会产生相同 residue key；
- 随机整数矩阵与暴力枚举结果一致。

### 结构映射

- reference 原子随机重排不改变 residue 集合；
- 每个 primitive site 恰好拥有 `abs(det(S))` 个 residue；
- 重复 residue 和缺失 residue 明确报错；
- 非对角和负元素超胞矩阵均可处理；
- source/target materialization 前后 atom lookup 一致。

### 回归

- 现有对角扩包的 orbit、位移数量和拟合参数数量不变；
- FC2、FC3、FC4 sparse IFC 往返键和值不变；
- ShengBTE、ALAMODE 和 phonopy 导出不改变已有安全案例结果；
- residue 层不调用 ASE MIC，不依赖 Wigner-Seitz；
- 大超胞不再使用固定 `[-1, 1]^3` 或未界定的浮点 residue。

## 实施顺序

1. 增加纯整数 3x3 determinant/adjugate 工具。
2. 将现有 `_translation_label()` 改为整数安全实现。
3. 实现 HNF 代表元枚举器，并与当前 BFS 结果做随机矩阵交叉验证。
4. 让 `PeriodicIndex` 使用新的 residue key 和代表表。
5. 保持用户 reference 直接映射路径，不强制重新生成代表元。
6. 增加非对角、负平移、原子重排和 alias collision 测试。
7. 检查 source/target materialization、SCPH/SSCHA 平移 lookup 和各 IO writer。
8. 删除不再需要的 BFS residue 枚举代码。
9. 更新周期索引、HNF residue 和结构转换文档。

## 非目标

- 不改变 cutoff 邻居策略；该内容属于 `PLAN.md` 的周期几何计划；
- 不实现 Wigner-Seitz 最近镜像；
- 不修改 FC2 旋转约束的物理公式；
- 不改变 HDF5 v2 的 lattice-labelled IFC schema；
- 不引入 phonopy 作为运行时依赖；
- 不对近似结构相似性做 residue 投影或静默修正。

## 与 cutoff neighbor context 的衔接研判

`CutoffNeighborContext` 的 ASE 邻居键包含完整的整数 offset，而 `PeriodicIndex` 的
`atom(primitive, translation)` 通过 residue 将它映射到 reference atom。对同一
`(primitive site, residue)`，多个 exact offset 可以合法地映射到同一个 reference atom；
这不是 residue 冲突，也不是 Wigner-Seitz 简并错误。

因此当前职责分层必须保持：

1. context 保存完整 `(i, j, offset)`，用于 cutoff 两侧稳定性和 alias multiplicity 诊断；
2. orbit 继续按 reference atom/residue 生成 cluster，保持现有 sparse IFC schema；
3. `PeriodicIndex` 只负责 exact integer translation 的 residue lookup；
4. MIC、全部简并最近像和联合 cluster 几何仍由 `PeriodicGeometry` 负责；
5. writer/materializer 在目标结构上重新解析需要的 exact offset，不从 cutoff context
   推断格式镜像。

### R0 实施门槛

纯整数 residue 替换可以先实施，且应满足：

- `_translation_label()` 与 `tools/supercell.py` 的实现统一；
- 不使用 `float(det)`, `inv`, `rint` 恢复伴随矩阵；
- 只改变 residue key 的计算，不改变代表元排序；
- 旧/新 key 在随机整数矩阵、负平移和非对角矩阵上逐项一致；
- `PeriodicIndex`、`build_supercell`、HDF5 v2 和 source/target materializer 回归通过。

R0 通过后才能把 residue key 用作 cutoff alias 的稳定诊断键。

### R0 执行记录（2026-08-20）

R0 已完成。新增独立的 `mlfcs._integer_lattice` 模块，提供：

- `determinant_3x3()`；
- `adjugate_3x3()`；
- `residue_key()`；
- `same_residue()`。

`core.geometry` 与独立的 `tools.supercell` 现在共享这组纯整数运算；工具模块没有依赖
MLFCS core。以下路径已替换浮点 determinant/inverse/rint residue 计算：

- `PeriodicIndex` 的 residue、完整性和 cell 数计算；
- `StructureRelation` 的超胞原子数验证；
- core 的 coset BFS label；
- `build_supercell` 的 coset label 和 determinant 验证。

已验证：

- 有符号非对角矩阵的整数 adjugate 双侧恒等式；
- 任意 `R + n @ S` 与 `R` 的 residue 等价性；
- tools/core 生成的 residue key 一致；
- 非对角负元素矩阵的 key 数量等于 `abs(det(S))`；
- 现有 supercell、reference mapping、cutoff 和拟合相关测试通过。

R0 没有改变代表元的 BFS 排序，因此没有改变现有 atom order；HNF 代表元替换仍属于
后续 R1/R2 评估。

### R1 alias 诊断

对每个 cutoff context 统计：

```text
alias(anchor_atom, target_atom) = number of stable integer offsets
```

该统计默认只写入诊断或日志，不改变 cluster 数量，不拒绝 `None` 或大 cutoff。若将来
用户要求 exact translation-labelled IFC，必须单独引入 R2 设计；不能通过修改
`PeriodicIndex.atom()` 或 `canonical_translation()` 静默实现。

### R1 执行记录（2026-08-20）

R1 已完成。`CutoffNeighborContext` 现在保留：

- 完整稳定 `ijS` offset 集合；
- alias atom-pair 数量；
- alias offset 总数；
- 单个 atom pair 的最大 offset multiplicity。

`InteractionSpace` 仅在存在 alias 时打印诊断，不改变候选 atom 集合、orbit、参数化或
IFC 支撑。

使用两个实际案例从原始 reference 和训练轨迹重新拟合，并与 R1 前已有 HDF5 结果逐项
比较：

| 案例 | orders | R1 alias 诊断 | 参数/残差 | sparse HDF5 key/tensor |
|---|---|---|---|---|
| SnSe 300 K | FC2/3/4 | FC2 自动 `None`：65536 pairs，最大 multiplicity 8 | 9440 joint params；train RMSE 0.01274950096，valid RMSE 0.01478947862 | FC2 2048、FC3 7280、FC4 6144；keys 和 tensors 完全一致 |
| Ba8Ga16Ge30 300 K | FC2/3 | 显式 5.4/4.35 Å，无 alias 日志 | 8329 joint params；train RMSE 0.02640398243 | FC2 1188、FC3 3222；keys 和 tensors 完全一致 |

SnSe 的 FC2 自动 cutoff alias 很大，但 R1 前后结果完全一致；这证明 alias 诊断没有
偷偷把 offset 拆成新的 IFC。Ba8Ga16Ge30 的复杂模型在 101 帧 NVE 数据上的严格拟合也
保持原有 215/382 orbits、1809/6520 参数和相同 sparse 输出。

因此 R1 通过，R2 仍未启动：当前 residue 折叠语义与实际拟合结果一致，尚无理由把 exact
offset 推入 orbit/IFC 参数空间。

### R2 下一步研判（拆分为影子评估与正式重构）

R2 不能直接作为当前 orbit 的一个布尔选项实现。实际原型表明，exact offset 需要同时
改变 cluster identity、对称 image、参数化 coordinates 和 sparse expansion；在 SnSe
这种 256 原子 reference、FC3/FC4 高阶组合中，直接复制现有 orbit 构造会产生不可接受的
重复枚举和内存开销。

因此 R2 分为两个子阶段：

#### R2a：shadow translation-labelled 评估（不进入核心）

在临时脚本或测试辅助模块中，对已有 sparse IFC 行执行：

1. 将 reference atom pair 转换成 primitive site + canonical residue；
2. 查询 `CutoffNeighborContext.keys` 中对应的 exact offset 数量；
3. 只统计潜在拆分行数、每阶支撑膨胀因子和重复 tensor 键；
4. 不生成新的 orbit、不改变拟合、不写入 HDF5。

R2a 的输出应是每阶的：

```text
existing sparse rows
potential exact-offset rows
unique exact physical keys
expansion factor
```

只有当某个实际用户场景确实需要区分这些 offset，才进入 R2b。

#### R2b：正式 exact cluster 重构（暂不实施）

正式重构必须先定义以下不变量：

- exact offset 是否是 primitive-lattice translation，还是 target-supercell offset；
- 周期等价 offset 是否聚合，聚合时 tensor 一致性如何验证；
- symmetry operation 如何变换 exact offset；
- ASR 是在 exact cluster 空间还是 residue quotient 空间施加；
- `translation_representatives` 是否仍表示一个 residue representative；
- ShengBTE/ALAMODE 的 Wigner-Seitz 解析是否从 exact cluster 重新开始。

在这些不变量确定前，不允许修改 `ClusterOrbit`、`OrderParameterization` 或 HDF5 schema。

#### 当前 R2 结论

R2a 是下一步可执行任务；R2b 不是局部优化，而是新的 IFC 数据模型。当前稳定且可复现的
默认路径继续使用 residue/target-atom 语义，完整 offset 只用于 context 诊断和 writer 的
目标结构解析。

### R2 exact cluster 的阻塞条件

把多个 offset 拆成多个 IFC 条目会改变的不只是 `PeriodicIndex`：

- orbit 的 canonical cluster 和 symmetry image；
- JAX coordinates 与参数数量；
- ASR 行空间；
- sparse expansion/materialization 的键；
- 有限差分位移和 fit snapshot 的 force 行语义；
- HDF5、ShengBTE、ALAMODE、SCPH/SSCHA 的平移相位。

所以 R2 不是本计划的收尾工作，而是新的 IFC 支撑设计。没有明确的跨模块键语义和完整
FC2/FC3/FC4 往返测试，不允许实施。

## ShengBTE 写出简化结论

hiphive 的 ShengBTE writer 不枚举完整 Wigner-Seitz 简并镜像。它先将 reference supercell 中的原子表示为：

```text
primitive site + integer offset
```

然后对 reference 中实际存在的 atom pair/triplet：

1. 计算当前 reference image 的实际 Cartesian 位移；
2. 使用 MIC 检查该实际位移是否已经是最近镜像；
3. 检查 cutoff；
4. 用 site label 和相对 offset 生成 ShengBTE key；
5. 对重复 key 聚合并验证 tensor 一致性。

因此 MLFCS 的 ShengBTE writer 可以采用同样的边界：

```text
source sparse IFC
    -> target reference supercell materialization
    -> 实际 atom/site/offset pair
    -> MIC 验证
    -> ShengBTE key
```

在当前 reference 已经明确、且 sparse IFC 已 materialize 到目标 supercell 时：

- residue 只用于 `site + offset -> target atom` 的查找，可以隐藏在 `PeriodicIndex` 中；
- 不需要 `joint_closest_image_shifts()`；
- 不需要枚举全部 Wigner-Seitz 简并镜像；
- 不需要对 ShengBTE 镜像做自动平均；
- 相同 key 只需聚合并检查数值一致性。

如果 source 和 target supercell 不同，显式 residue 仍然属于 materializer 的职责，用于 source translation 到 target atom 的映射；它不意味着 ShengBTE writer 必须重新进行 Wigner-Seitz 枚举。

这条路线保留 ASE MIC 的一般晶胞验证能力，同时让 ShengBTE 的输出语义与 reference supercell 原子/offset 表示一致，避免因联合最近镜像组合造成块数量、平移代表和聚合方式的不必要差异。

## ALAMODE 写出简化策略

ALAMODE 的 27-image 限制是格式语义，但不要求 writer 对每个 IFC 条目重新执行通用 Wigner-Seitz 搜索。目标策略是把格式编码和通用几何搜索分离：

```text
target ExportView
    -> source/target residue materialization
    -> target atom index
    -> 一次构造 ALAMODE 27-image mapping table
    -> FC2/FC3/FC4 XML writer 查表写出
```

`AlamodeImageTable` 在一个 target supercell 上只构造一次，保存：

- 27 个固定 mirror translation 与 mirror id 的对应；
- target atom pair 到合法 mirror id 的映射；
- 多个等距合法镜像时的分配权重；
- 一次性的 general-MIC 可表示性验证。

后续写出 FC2、FC3、FC4 时只查表，不在每个 IFC 条目中重新创建 `PeriodicGeometry`、调用 `find_mic()` 或枚举通用 Wigner-Seitz 镜像。

以下限制仍然保留：

- 若真实最近镜像不在 ALAMODE 固定的 27 个镜像中，目标结构必须先做等价的整数晶格换基，仍不可表示时拒绝写出；
- 如果一个物理项对应多个等距镜像，按 ALAMODE 语义写出多个条目并使用预先确定的权重；
- FC3/FC4 中重复 atom 的镜像选择必须保持一致；
- source/target residue 映射属于 `ExportView`/materializer，不由 XML writer 自己实现。

### R2a 后续审计执行状态

R2a 的结论是：不实施 exact cluster 拆分。当前实现只把完整 offset 图作为 orbit 的唯一
支撑来源，并缓存一条确定性 pair vector；`atom_neighbors` 仍作为兼容性诊断字段保留，
但不再参与 orbit 候选生成。等距最近镜像不被视为错误 alias，非等距 image 才会在显式
cutoff 下拒绝。这样删除了旧的联合 Wigner 组合和 `1e-2 Å²` 二次过滤，同时不改变 IFC
schema 和高阶参数空间。

该策略不改变 ALAMODE 的输出语义，只把当前逐条镜像搜索改为一次性有限表查找，减少高阶写出中的重复 MIC 和组合开销。
