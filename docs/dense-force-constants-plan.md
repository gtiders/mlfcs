# 统一稠密力常数表示与长程 FC2 接入方案

本文是后续实现方案，不表示这些接口已经存在。本阶段只新增本文档，不修改生产代码或原生存储格式；已有整数审查、ASR 与旋转研究结论保持不变。

## 1. 结论与设计范围

方案可行：原生 HDF5 继续保存参数化 `ForceConstants`；计算和导出时，构造一个按需提供完整 Cartesian 张量的 `DenseForceConstants`。它既能读取参数化模型的短程贡献，也能叠加 Ewald 提供的长程 FC2。

这里“稠密”描述输出张量的定义：任意指定的有限超胞原子组合都有一个完整的 Cartesian 张量，包括零块。它不要求构造时物化全部数组，也不要求枚举所有零块才能完成内部计算。

统一的是张量数值、单位、索引约定与分块读取接口。以下两类信息不能混同：

1. 未折叠的原胞晶格张量：保留每个周期镜像的相对平移。
2. 有限周期超胞张量：同一超胞原子组合的镜像贡献已经相加。

后者不能一般地恢复前者。因此统一模型必须保留尚可用的晶格来源，不能把超胞稠密数组当成所有格式都可逆转换的唯一存储。

本方案不实现 Ewald 求和公式，也不假定 Ewald 是固定电荷模型还是 Born 有效电荷模型。接入层只接收与本项目定义一致的能量 Hessian；两类电荷模型的物理定义不能互换。

## 2. 当前代码与需要改变的位置

| 当前模块 | 实际职责 | 需要调整的方向 |
|---|---|---|
| `force_constants/model.py` | 参数向量、原生 save/load、导出入口 | 保持参数化模型；`write` 委托统一张量表示 |
| `force_constants/native.py` | HDF5 v5 的模型与系数存储 | 本轮不变，不保存派生稠密数组 |
| `force_constants/expansion.py` | 逐轨道求值，但最终收集全部 lattice 张量 | 抽取唯一的按需晶格块求值逻辑 |
| `force_constants/export.py` | 全量 compact 折叠、FC2 全超胞展开、阈值和索引转换 | 张量变换移入统一表示；格式相关清理保留在导出层 |
| `force_constants/formats.py` | 格式调度与文本/HDF5 编码 | 只消费统一表示提供的数值块和必要标签 |
| `phonon/dynamics.py` | 未折叠 FC2 的 Fourier 求和与质量加权 | 不改为直接使用折叠矩阵计算任意 q 点 |
| `phonon/scph.py` | 当前会展开 FC4，并依赖参数模型 | 不在导出迁移中同步改算法 |

当前 Phonopy 文本会同时物化 compact、full 数组和全部文本；HDF5 虽按首原子写入，仍先物化 compact 数组；ShengBTE 会收集全部张量与文本。新表示要消除这些无必要的全量中间结果。

`expansion.py` 当前还有声子和测试消费者。迁移时保留其现有返回接口作为新求值器的收集适配器；它不得维护第二份旋转、置换或参数遍历算法。此处保留的是仍有消费者的接口，不是旧文件路径兼容层。

## 3. 力常数的统一数学定义

力常数始终是能量导数，不是力的 Jacobian，也不是质量加权动力学矩阵：

$$
\Phi^{(p)}_{a_1\alpha_1,\ldots,a_p\alpha_p}
=\frac{\partial^p E}
{\partial u_{a_1\alpha_1}\cdots\partial u_{a_p\alpha_p}}.
$$

- 能量单位 eV，位移单位 Å；第 $p$ 阶力常数单位 eV/Å$^p$。
- 二阶 Hessian 满足 $H=-\partial F/\partial u$。
- Taylor 能量中的 $1/p!$ 不包含在张量值中。
- 不额外乘轨道大小、排列数量或重复原子的组合因子。
- 原子轴按张量索引顺序排列；Cartesian 轴依次为 x、y、z。
- 未提供的阶数仍表示“模型没有这一阶”，不能自动当成全零阶。
- 已提供的短程阶数中，截断外的 lattice 组合为零；长程贡献不受短程 cutoff 限制。

在每个轨道中，先求代表张量，再按当前 symmetry operation 和 axis permutation 产生镜像张量：

$$
\Phi_{o,\mathrm{rep}}
=\operatorname{reshape}_{C}(B_o c_o,(3,\ldots,3)).
$$

这里 $B_o$ 是现有 `component_basis`，$c_o$ 是现有物理参数。继续使用已验证的 Cartesian 旋转及置换约定，不在本次转换中重新定义 lattice/Cartesian 关系。

## 4. 几何、标签与数组形状

记原胞原子数为 $N$，超胞原子数为 $n$，当前张量阶数为 $p$，一批晶格块数为 $b$。

### 4.1 几何约定

晶胞矩阵的三个晶格向量为行；分数坐标和晶格平移也为行：

$$
x=sA,\qquad A_{\mathrm{super}}=SA,\qquad
n=N|\det S|.
$$

`ClusterMap` 保持用户输入超胞的原子顺序，不假定 ASE repeat 顺序。非对角 $S$ 继续使用现有商群映射，不改为各坐标独立取模。

| 名称 | 形状 | 单位/含义 | 来源 |
|---|---|---|---|
| primitive cell $A$ | `(3, 3)` | Å，行晶格向量 | `ClusterSpace.cell` |
| primitive fractional positions | `(N, 3)` | 原胞分数坐标 | `scaled_positions` |
| primitive atomic numbers | `(N,)` | 元素编号 | `atomic_numbers` |
| primitive masses | `(N,)` | amu；不混入力常数 | `masses` |
| supercell matrix $S$ | `(3, 3)` | 原胞到超胞的整数矩阵 | `ClusterMap` |
| supercell site indices $i_a$ | `(n,)` | 每个超胞原子对应的原胞 site | `primitive_site_indices` |
| supercell translations $t_a$ | `(n, 3)` | 原胞晶格平移 | `lattice_translations` |
| supercell cell/positions/species | `(3,3)` / `(n,3)` / `(n,)` | 明确外部原子顺序 | `ClusterMap` |

这些数据优先引用现有域对象的数组，不在 DenseForceConstants 中再复制一套几何。质量只由需要动力学矩阵的消费者读取。

### 4.2 未折叠晶格块

第一原子位于中心原胞。每批返回三组数组：

| 数组 | 形状 | 定义 |
|---|---|---|
| `sites` | `(b, p)` | 有序原胞 site 编号 |
| `translations` | `(b, p-1, 3)` | 后 $p-1$ 个原子相对第一原胞的平移；第一平移隐含为零 |
| `tensors` | `(b,) + (3,)*p` | 未折叠完整 Cartesian 张量 |

张量方向按 C order 排列。该批大小由工作区预算决定，不随总镜像数增长。

ShengBTE 写出的向量是 $R_kA$，不包含原胞内部位置差。计算实际间距时才使用：

$$
r_{ij}(R)=(s_j-s_i+R)A.
$$

镜像标签必须保留整数平移，不能用 minimum-image 距离替代。

### 4.3 compact 超胞表达

以第一原子为中心原胞 site，后续原子使用目标超胞索引：

$$
C_p\text{ 的形状}=(N,\underbrace{n,\ldots,n}_{p-1},
\underbrace{3,\ldots,3}_{p}).
$$

| 阶数 | compact shape |
|---|---|
| FC2 | `(N, n, 3, 3)` |
| FC3 | `(N, n, n, 3, 3, 3)` |
| FC4 | `(N, n, n, n, 3, 3, 3, 3)` |

这是统一有限超胞数值接口的主表示，Ewald 的 FC2 也必须归一到这里。

### 4.4 full 超胞表达

$$
F_p\text{ 的形状}=(\underbrace{n,\ldots,n}_{p},
\underbrace{3,\ldots,3}_{p}).
$$

full 表达仍是按需视图，不自动申请完整 ndarray。需要真正数组时使用显式物化方法。

compact 占用 $8Nn^{p-1}3^p$ 字节，full 占用 $8n^p3^p$ 字节，两者相差 $n/N$ 倍。对于 $N=2,n=250$ 的 FC3，分别约为 27 MB 和 3.375 GB；因此分块必须发生在物化之前。

## 5. lattice → compact → full 的转换

### 5.1 折叠

将每个后续标签 $(i_k,R_k)$ 映射为超胞原子 $a_k$，并累加：

$$
C_p[i_1,a_2,\ldots,a_p]
\mathrel{+}=\Phi^{(p)}(i_1,\ldots,i_p;R_2,\ldots,R_p).
$$

两个不同平移落入相同商群类时，它们是 alias，贡献相加。一个来源中的相同未折叠标签不能重复计入；多个明确的物理来源则可以在相同标签或折叠块上叠加。

实现可预先按输出原子组合分组镜像索引。保存的是标签、偏移和来源编号，不是全部求值张量。每个目标块只访问其相关贡献，不能为每个输出块重新扫描全部 orbit images。

### 5.2 平移展开

令 $\pi_a(b)$ 为将超胞原子 $b$ 的原胞平移减去第一原子 $a$ 的平移后，再映射到目标超胞的索引：

$$
\pi_a(b)=\operatorname{atom\_index}(i_b,t_b-t_a).
$$

则：

$$
F_p[a_1,\ldots,a_p]
=C_p[i_{a_1},\pi_{a_1}(a_2),\ldots,\pi_{a_1}(a_p)].
$$

这一步只改变原子索引，不旋转 Cartesian 轴，也不再次对 aliases 求和。它复用现有 `translated_atom_indices` 的数学定义。

compact 的第一索引始终表示平移零处的 primitive site，不是“该元素在超胞中第一次出现的原子”。从 full 转回 compact 时，选择商群标签为零的原子 $a_0(i)$：

$$
C_p[i,b_2,\ldots,b_p]=F_p[a_0(i),b_2,\ldots,b_p].
$$

如果来源采用了其他 anchor，必须先做对应索引重排。不能直接截取 full 的前 $N$ 行。

### 5.3 不可逆的信息

compact/full 相互转换只在 primitive 平移不变的有限周期模型内成立；缺陷超胞的任意 Hessian 不属于这个契约。

折叠会丢失独立镜像的贡献分配。不能把 $C_p$ 拆给任意选定的平移，也不能通过“平均分配给最短镜像”声称恢复原模型。未折叠标签只有在来源保留或明确提供它们时才存在。

## 6. Ewald FC2 数组的接入

### 6.1 从超胞能量 Hessian 转换

如果 Ewald 输出 atom-major Hessian：

$$
H\text{ 的形状}=(3n,3n),\qquad
H_{3a+\alpha,3b+\beta}=F_2[a,b,\alpha,\beta],
$$

必须先换轴，不能直接 reshape 为 `(n,n,3,3)`：

```python
full_fc2 = hessian.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
hessian = full_fc2.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
```

再按上一节的中心原胞 anchor 提取 `(N,n,3,3)`。若 Ewald 原子顺序与 ClusterMap 不同，设 `source_indices[a]` 是目标原子在 Ewald 数组中的位置，两个原子轴都必须重排：

```python
aligned = full_fc2[np.ix_(source_indices, source_indices)]
```

若来源采用 axis-major `(alpha,a,beta,b)`，先由适配器显式转换为 atom-major。来源若是力 Jacobian，先取负；来源若质量加权，则不能当 Hessian 直接相加。格式不能通过猜测数组形状来识别。

### 6.2 接入时的唯一数值布局

第一阶段只接受归一化后的额外 FC2：

```text
additional_fc2: float64, shape (N, n, 3, 3), eV/angstrom**2
```

数组表示同一目标周期超胞的完整长程贡献，使用相同原子顺序、Cartesian 基、anchor 和电边界条件。它必须包含相应 self/onsite 项，不能仅提供非 onsite 相互作用。

数组归一化由 Ewald 适配器完成，统一模型不通过多个 shape 分支猜测 full、compact、Hessian 或 dynamical matrix。

未来 Ewald 若能直接生成 compact 分块，内部求值器改为从 Ewald 读取当前块；对写出层仍提供同一块接口。不能为了接入统一模型先强制 Ewald 申请完整超胞 Hessian。

### 6.3 相加

$$
C_2^{\mathrm{total}}=C_2^{\mathrm{parameterized}}
+C_2^{\mathrm{additional}}.
$$

- 在相同目标块内相加，然后才执行导出阈值。
- 不要求长程项落在短程 ClusterSpace 的参数列空间内。
- 不把长程矩阵拟合回截断参数模型后再导出。
- FC3/FC4 没有额外来源时继续使用参数模型；FC2 的附加项不能产生隐含的高阶贡献。
- 只有附加项也具有未折叠来源时，才可以提供总 FC2 的未折叠晶格块。
- 总 FC2 只有 folded 来源时，晶格遍历必须明确报错，不能悄悄只返回短程部分。

Ewald 非线性能量一般也有更高阶导数；本接口仅接入 FC2 不等于宣称该能量的全部高阶长程导数为零。谐波偶极近似与完整固定电荷势的截断必须由物理生产者说明。

## 7. 拟议接口与生命周期

公共 API 保持直接类型初始化，不引入 `build` 或 `from_*` 链：

```python
dense = DenseForceConstants(force_constants, cluster_map)

# 用户确认参数模型尚未包含这个附加项时，显式叠加。
dense_total = DenseForceConstants(
    force_constants,
    cluster_map,
    additional_fc2=long_range_compact_fc2,
)

dense_total.write("force_constants.hdf5", format="phonopy", order=2, storage="hdf5")
dense_total.write("FORCE_CONSTANTS_3RD", format="shengbte", order=3)
```

`cluster_map` 可省略，此时仅提供原胞晶格遍历及 ShengBTE/TDEP 写出。访问 compact/full 或叠加 folded FC2 必须提供 map；不创建假的默认超胞。

| 接口 | 返回与语义 |
|---|---|
| `orders` | 参数模型明确具有的阶数；本阶段附加 FC2 要求基础模型也有 FC2 |
| `lattice_blocks(order, *, max_bytes=8*1024**2)` | 按确定标签顺序返回 `(sites, translations, tensors)` 批次；不含截断外零标签 |
| `compact_blocks(order, *, max_bytes=8*1024**2)` | 返回 `(atom_slices, tensors)`；前者为 $p$ 个原子轴的 slice，Cartesian 轴始终完整 |
| `full_blocks(order, *, max_bytes=8*1024**2)` | 同上，第一原子轴使用超胞索引；包含完整有限域的零值语义 |
| `compact_array(order)` | 显式物化 compact ndarray |
| `full_array(order)` | 显式物化 full ndarray |
| `write(...)` | 使用已有格式、order、storage、threshold 语义 |

若一个 compact 块的原子轴宽度为 $w_1,\ldots,w_p$，返回张量形状必须是：

$$
(w_1,\ldots,w_p,\underbrace{3,\ldots,3}_{p}).
$$

切片只分割原子轴，不能返回一半 Cartesian 张量。预算至少要容纳一个 $3^p$ 输出张量及本次算法必须同时存活的暂存数组；不足时在申请之前报告，不能只检查输出字节数。

`max_bytes` 约束求值器拥有的同时存活数值工作区，包括输出块、附加项暂存和代表张量；不包含调用者已有的模型、附加数组或消费者保留的返回块。生产导出循环只持有当前批次。索引表开销单独统计，不能把它宣称为零内存。

`compact_array` 和 `full_array` 明确申请完整结果，该结果不受流式工作区预算限制，但其 shape、字节数和平台可分配范围必须在物化入口检查。内存报告同时列出最终数组与暂存，不能以“内部使用分块”掩盖显式物化的总占用。

### 所有权

- 参数模型和 map 按引用持有，不重复复制几何、orbit 基或系数。
- 外部可写 `additional_fc2` 在构造时复制一次形成稳定快照；已符合契约的连续只读数组可共享。
- 生成器返回的数值批次不会被下一次迭代覆盖；消费者若主动收集所有批次，自行承担全量内存。
- 不引入内容 fingerprint、跨对象签名或每块重复兼容性验证；用户负责匹配几何、原子顺序和模型物理含义。
- 数组形状、有限值、预算及必要整数索引范围在各入口检查，不携带跨对象准入证书。
- 不缓存整个阶数的稠密结果，不引入全局可变工作区；每次迭代拥有自己的局部工作区。

## 8. 导出层规则

| 格式 | 读取的表达 | 处理方式 |
|---|---|---|
| Phonopy FC2 文本 | full 分块 | 按原子对写入，包括零块 |
| Phonopy FC2 HDF5 | full 分块 | 保持当前完整超胞 shape 和 dataset 名 |
| Phono3py FC3 HDF5 | full 分块 | 保持当前完整超胞 shape 和 dataset 名 |
| ShengBTE FC3 / FC4 扩展文本 | 未折叠 lattice 块 | 保留 primitive site、相对平移；筛除全零块 |
| TDEP FC2/FC3/FC4 | 未折叠 lattice 块 | 保持原胞分组及相应文本规则 |
| 原生 HDF5 save/load | 参数模型 | 不经过 DenseForceConstants |

Phonopy 与 Phono3py 也支持 compact 文件，但本轮不增加文件布局选项，避免把内部统一表示迁移与外部 API 改动绑定。[Phonopy 格式](https://phonopy.github.io/phonopy/input-files.html#force-constants-and-force-constants-hdf5)、[Phono3py 格式](https://phonopy.github.io/phono3py/input-output-files.html)。

### 流式写出细节

1. Phonopy 文本逐块输出，删除全量 `lines` 和 full ndarray 中间结果。
2. HDF5 同时分割多个原子轴，不仅分割第一个轴；FC3 的单个首原子切片仍可能过大。
3. HDF5 dataset 的 chunk 大小遵守写出预算，不再固定为整个首原子切片。
4. ShengBTE 头部需要非零块数：第一遍按相同阈值计数，第二遍重建并流式输出。允许两次求值，不保存全部数值张量和文本。
5. 晶格索引与排序可以预处理一次，后续两遍复用；数值不需要全量缓存。
6. TDEP 同样先获取各原胞原子的块数，再逐组输出；保持现有保留零标签块的行为。
7. 短程 aliases 和附加贡献全部累加后，才将绝对值小于 threshold 的分量置零。统一张量读取本身不做阈值清理。
8. 不重新对 lattice 块套排列倍数；保持当前 ordered tensor 与 anchor 定义。
9. ShengBTE 不再要求仅用于检查、实际未使用的 ClusterMap。
10. 只有 folded 附加 FC2 时，TDEP 的总 FC2 晶格导出不支持；显式导出基础 ForceConstants 可得到基础模型，不能通过默认行为遗漏长程项。

`ForceConstants.write(...)` 仍可使用：内部构造不带附加项的 DenseForceConstants，再走同一格式实现。格式 writer 不再直接遍历参数、读取 orbit basis 或执行旋转。

## 9. 原生保存什么，运行时保留什么

### 9.1 当前原生 HDF5 v5

继续保存已有参数化数据：

- 原胞 cell、scaled positions、atomic numbers、masses；
- symmetry rotations、translations、site permutations、site shifts、Cartesian rotations；
- 阶数、cutoff、max body order、orbit/parameter offsets；
- orbit 标签、操作、置换、lattice/component basis、observation rows；
- 每个已有阶数的物理系数向量。

不保存 DenseForceConstants、目标超胞派生索引、求值缓存或展开张量。`DenseForceConstants` 不增加 save/load 或 pickle 接口。

### 9.2 稠密运行时对象必须持有

| 数据 | 是否必须 | 是否写入当前原生文件 |
|---|---|---|
| 参数模型引用 | 是 | 模型自身已保存 |
| 明确的阶数与单位约定 | 是 | 阶数来自模型，单位由格式定义 |
| ClusterMap 引用 | compact/full 操作必须 | 否 |
| lattice 标签及索引分组 | 晶格来源/折叠时需要，可惰性准备 | 否 |
| 附加 FC2 数组或未来 Ewald 求值器 | 有附加贡献时必须 | 否 |
| 附加贡献已包含于总 FC2 的说明 | 有附加贡献时必须，供报告解释 | 否 |
| 全量 compact/full 数组 | 仅显式物化时 | 否 |
| Fourier/电边界描述 | 使用长程求值器时必须 | 当前未实现 |

保存基础参数模型不会保存用户叠加的外部 FC2。当前完整工作流必须保留生成 Ewald 数据的输入，并在 load 后重新接入，不能声称 load 恢复了 combined 模型。

### 9.3 未来 Ewald 可复现输入

后续若要让原生文件恢复“短程参数模型 + 长程物理模型”，应另行版本化扩展存储输入配方，而不是把整个稠密矩阵塞入系数向量：

- charge 模型种类及数学定义；
- 固定电荷 `(N,)`，单位为基本电荷 $e$，或 Born charge tensors `(N,3,3)`，按模型选择其一；Born 张量约定第一 Cartesian 轴为极化方向、第二轴为位移方向，$Z^*_{i,\alpha\beta}=\Omega\,\partial P_\alpha/\partial u_{i\beta}$，数值以 $e$ 为单位；
- 必要的 Cartesian dielectric tensor `(3,3)`，使用无量纲相对介电常数，并说明是该模型使用的哪一种介电响应；
- 电边界条件、零 reciprocal vector / self term 的处理规则；
- 求和收敛控制、Ewald 参数和算法定义版本；若使用 splitting 参数，明确其量纲为 Å$^{-1}$，实空间与 reciprocal cutoff 分别为 Å 和 Å$^{-1}$，不能只保存没有单位的 `alpha`/`cutoff`；
- 基础参数系数是 short-range residual 还是已经包含长程项；
- 若采用长期绑定超胞的配方，保存对应 $S$ 和原子标签；若是原胞物理模型，则超胞仍由运行时指定。

这不是本轮 HDF5 v5 的隐式增量修改。只有收到对应功能任务后才设计新版本；任意外部数组没有生成配方时，不能承诺仅靠参数化文件恢复它。

长程生产者还须说明其电中性条件：固定电荷的原胞电荷和，或 Born 张量的逐分量和。是否允许背景补偿、如何处理电中性偏差属于电模型的定义；统一稠密层不自动更改电荷。电荷到 eV/Å² 的 Coulomb 单位换算在生产者中完成一次，写出层不再乘单位系数。

## 10. 长程物理边界与防止重复计入

### 10.1 训练数据的分解

只有参数模型尚未包含附加长程贡献，才可以直接相加。若拟合已经消费了总力，再无条件加 Ewald 就会重复计入。

长程/短程分解后的训练应满足：

$$
F_{\mathrm{fit}}=F_{\mathrm{reference}}-F_{\mathrm{long}},\qquad
\Phi^{(2)}_{\mathrm{total}}=\Phi^{(2)}_{\mathrm{fit}}+\Phi^{(2)}_{\mathrm{long}}.
$$

减去的力和加回的 Hessian 必须来自同一长程定义与电边界条件。若只采用谐波长程模型，则其力应按相同参考几何的线性近似计算；完整 Ewald 力和仅 FC2 Hessian 的截断关系必须说明。

统一数组接口不自动判断训练数据的物理来源，也不自动从总力常数中“猜出”长程部分。`additional_fc2` 的明确传入即表示用户确认允许叠加。

### 10.2 ASR 与对称性

叠加前应分别确认基础模型和长程项的定义。具有平移不变性的 long-range Hessian 包含与非 onsite 项配套的 onsite/self 项；两项均满足 ASR 时，总和也满足 ASR，浮点误差另行诊断。

不能调用参数模型的 ASR/旋转方法，把总长程矩阵强行投影回截断 ClusterSpace。当前后处理 API 仍只操作参数模型；未来若对 dense 总模型投影，需要独立定义其修正度量和物理电边界。

### 10.3 有限超胞不等于完整长程 q 依赖

周期超胞长程 Hessian 是在指定 Ewald 求和及电边界下的有限周期表达。形式上，其 compact 值包含超胞晶格平移对应的镜像贡献；对于条件收敛的相互作用，这种求和必须采用生产者规定的 Ewald 定义，不能用任意顺序的直接求和代替。

与超胞相容的分数倒空间点满足 $Sq^T$ 为整数列。在相同 Fourier 与电边界约定下，周期模型可以在这些点与其生产者的有限周期响应对照。一个普通有限稠密矩阵不能普遍恢复任意 q 的无限长程响应或极性材料方向相关的 $q\to0$ 极限。[Phonopy 非解析项与长程处理说明](https://phonopy.github.io/phonopy/formulation.html#non-analytical-term-correction)。

现有声子相位约定保持为：

$$
D_{i\alpha,j\beta}(q)
=\frac{1}{\sqrt{m_i m_j}}
\sum_R\Phi_{i\alpha,j\beta}(R)
\exp\left[2\pi\mathrm{i}\,q\cdot(R+s_j-s_i)\right].
$$

原胞倒空间动力学矩阵形状为 `(nq,3*N,3*N)`，一般为 complex128；它与 `(N,n,3,3)` 的实数超胞力常数不是同一个对象。Cartesian reciprocal wave vector 为 $2\pi qA^{-T}$，不能把分数 q 直接当 Å$^{-1}$ 输入。

因此下一阶段长程声子应保留“短程晶格 Fourier 项 + 长程 reciprocal evaluator”的物理入口，或者明确采用有限超胞插值近似。禁止从 folded Ewald FC2 随意补一组 R 后冒充无限晶格项。

如果外部 Phonopy 流程还启用 BORN/NAC，需要与其长短程分解方案一致。不能把已有 long-range FC2 再无条件加一遍相同贡献；统一稠密导出本身不自动生成 NAC，也不宣称处理了这个分解问题。

## 11. 模块与迁移顺序

建议新增一个领域文件 `force_constants/dense.py`，包含 DenseForceConstants 和它专用的求值、分块、折叠辅助函数。不要拆成大量 provider、buffer、view、manager 小模块。

1. **冻结现有导出**：保存同一模型在四种格式的临时数值参考，覆盖 FC2/3/4、aliases、原子重排和非对角超胞。
2. **统一求值**：实现唯一的按需 lattice 块生成；现有 `expand_lattice_tensors` 通过收集它保持声子消费者接口。
3. **实现 DenseForceConstants**：compact/full 分块与显式物化；此时不接 Ewald，也不修改拟合和声子算法。
4. **迁移格式 writer**：逐个改成读取统一表示，保持数据集、单位、阈值和顺序；删除被替代的全量导出算法。
5. **接入额外 compact FC2**：先用已知数组验证分块相加与布局转换，不实现 Ewald 公式。
6. **清理职责**：将所有张量展开/折叠移出格式层；无剩余消费者后合并或删除 `export.py` 的对应 helper，不留下第二套求值实现。
7. **后续独立任务**：选择具体 Ewald 电模型、力分解、reciprocal 长程求值与存储版本扩展。不能在前面步骤中偷偷加入一种默认电模型。

本轮不把 high-level 拟合 API、SCPH、ASR/旋转投影或原生文件格式一起重写。

## 12. 验收要求

### 数值与数组变换

- 逐块求值与现有 orbit expansion 一致；包含非平凡旋转和轴置换。
- 晶格 alias 先求和，再阈值清理；覆盖相消和两个小分量相加超过阈值的情况。
- 对角/非对角 $S$、超胞原子乱序以及非零 anchor 下，compact/full 转换正确。
- 用可区分所有索引的 `(3*n,3*n)` 数组验证 reshape/transpose，避免形状正确但方向轴错位。
- 有 primitive 周期性的 full Hessian 正确提取 compact；任意缺陷 Hessian不宣称可压缩。
- 基础 FC2 与附加 FC2 的逐块结果等于独立物化后相加；包含短程 cutoff 外的附加非零块。
- FC2 叠加不改变 FC3/FC4；缺失阶数继续报错。
- 只有 folded 来源时，不允许总 FC2 的晶格导出，不能漏掉附加项。

### 格式与内存

- Phonopy/Phono3py 文件用对应 reader 验证 shape、原子顺序与数值。
- ShengBTE 检查平移向量、块计数、Cartesian 顺序、非零块筛选以及两遍结果一致性。
- TDEP 保持其 per-atom 块数、零块及 FC2 polar flag 的已有行为；不伪装输出未实现的长程 polar 元数据。
- 原生 v5 save/load 结果与布局不变；DenseForceConstants 不参与序列化。
- 测量写出端峰值内存，确认没有随完整 $n^p3^p$ 数组规模增长的常驻数值结果。
- 分开报告冷 JIT、热运行、索引准备、张量求值和磁盘编码耗时；流式不会减少目标文件本身的大小。
- 记录附加输入数组的内存，不能把调用者预先物化的 Ewald Hessian 排除后宣称整个流程只占一个块。

### 将来的 Ewald 验收

- 与相同单位、原子顺序和电边界下的独立能量/力导数比较 Hessian。
- 检查有限周期 Hessian 的互易对称性、ASR 与 onsite/self 一致性。
- 在相容 q 点比较有限周期模型和生产者响应；任意 q/NAC 使用独立长程验证。
- 证明训练扣除与导出加回同源，避免重复计入。

所有新增类、函数和私有成员按 AGENTS.md 编写英文 docstring：先说明力常数对象及其边界，再说明索引、单位、形状和不可逆转换；不以数组存储细节代替物理定义。


## 补充：ForceDataset 与已选定的长程实现

保留以上稠密接口方案作为基线。当前新增 `mlfcs.dataset`，采用三维 Born 偶极 Ewald 加现有非 onsite FC2 orbit 上的局域修正。长程部分自身满足 ASR 与 Hessian 对称性。`Ewald.compact_fc2` 和 `full_fc2()` 输出数值数组，尚未实现本方案中的统一稠密 writer。HDF5 仍只保存参数化模型；详见[数据集与 Ewald](dataset-api.md)。

## 接口修订：独立 Ewald 与统一 compact 张量

本节记录后续接口设计，保留前文作为基线；以下接口尚未实现。前文的 `DenseForceConstants` 名称及 `additional_fc2` 接入草案由本节替代。统一张量类型不识别 Ewald 对象，也不承担 Ewald 求和。

### Ewald 的生命周期

```python
ewald = Ewald(
    cluster_space,
    born_charges=born,
    dielectric=epsilon,
    rtol=1e-8,
    atol=1e-10,
)

long_forces = ewald.forces(mapping, data.displacements)
long_fc = ewald.force_constants(mapping)
```

初始化绑定原胞物理模型与 FC2 局域修正支持空间，不绑定任何超胞。初始化计算原胞层的 Ewald 行和、局域修正及 onsite；Born/介电张量、电边界和求和容差也在此固定。继续接受 ClusterSpace，是因为修正支持空间来自其非 onsite FC2 orbit，而不是因为长程尾受该 cutoff 截断。

每次 `forces(mapping, displacements)` 或 `force_constants(mapping)` 都为显式指定的目标超胞计算周期 compact FC2，不更换 Ewald 自身的当前 mapping，不维护隐式全局缓存。调用前者时，一批位移共享一次求和；不是每帧重新求和。两个入口调用同一内部求值流程，力始终是同一 compact Hessian 的负收缩。

`displacements` 单帧形状为 `(n, 3)`，多帧为 `(frames, n, 3)`，单位 Å，顺序为指定 mapping 的 supercell_atoms；力返回相同形状，单位 eV/Å。它仍是固定 Born/介电张量下的谐波模型，不会根据每帧结构更新电学数据。

换 mapping 可复算，但其 primitive geometry、site 顺序与模型必须相容，由调用者负责配对；不引入 fingerprint 或新准入证书。形状、有限值和必要的数值范围检查保留。改变局域修正 cutoff 则构造新的 Ewald 模型；仅换目标超胞不改变原胞层修正。

### 同一张量类型与组合

```python
short = CompactForceConstants(short_model, mapping)
long = ewald.force_constants(mapping)
total = short + long

compact = total.compact_array(order=2)  # (N, n, 3, 3)
total.write("force_constants.hdf5", format="phonopy", order=2, storage="hdf5")
```

短程转换与 Ewald 输出均为 `CompactForceConstants`。Ewald 输出仅有 FC2，含 self/onsite 和修正；它保存已计算的 compact 张量，不持有 Ewald 对象、求和回调或电学参数。张量类型只依赖参数模型、通用张量数据与 mapping，不能反向导入 dataset/ewald.py。

`long.harmonic_forces(displacements)` 可复用已生成的 FC2，供多次力计算与导出共用；此方法只收缩 FC2，不暗示计算 FC3/FC4 的非线性力。`ewald.forces(mapping, displacements)` 使用同一通用收缩实现，不维护第二份力算法。

短程表示按需求值，长程表示引用自身持有的 compact 数值结果；总表示按块相加，不提前物化 full 或总 compact 数组。相加要求同一目标超胞、原子顺序及单位，由调用者负责语义配对；不会推断或自动换算。阶数取两个明确来源的并集：Ewald 只对 FC2 贡献，原有 FC3/FC4 不变。阈值清理只发生在相加后的格式写出阶段。

未叠加长程项的 `ForceConstants.write` 委托同一个张量及格式实现；不保留第二套 exporter。原生 save/load 继续只保存参数化模型，不保存总表示或 Ewald 配方。原 Ewald 的 mapping 字段、无 mapping 的 forces、compact_fc2 属性和 full_fc2 方法在这次 API 迁移中移除；数组读取统一由返回张量对象承担。

### 未折叠与 folded 的区别

未折叠值 $\Phi_{ij}(R)$ 分别记录每个 primitive site 与每个相对晶格平移的贡献。folded compact 则记录目标超胞周期等价类内的总和：

$$
C_{i,a}=\sum_{R:\operatorname{map}(j,R)=a}\Phi_{ij}(R),
$$

长程求和按 Ewald 指定的边界和收敛定义解释，而不是任意顺序的直接求和。

用一维三倍超胞说明标签信息：平移 $1,4,-2$ 都落到同一超胞原子。某个张量分量若分别为 $2,3,5$，compact 只保存 $10$。从 $10$ 无法恢复三份贡献，因为 $1,9,0$ 也会给出同一个结果。三维非对角超胞遵循相同的周期商群规则，而非逐坐标取模。

短程参数模型还保存 orbit 与各镜像标签，所以可独立求出这些贡献。Ewald 输出的 compact 是指定超胞下的数值结果；没有携带其原胞物理配方，因而不能恢复单独镜像贡献或任意 q 的无限长程响应。原始 Ewald 对象仍保存物理定义，这不等于其导出的张量自动携带该定义。

这不是两个张量类型，而是同一种张量类型中实际可用的信息不同：

| 操作 | 短程张量 | Ewald 的 folded FC2 | 两者相加后的 FC2 |
|---|---|---|---|
| compact/full 块与显式数组 | 支持 | 支持 | 支持 |
| 固定目标超胞 FC2 力收缩 | 支持 | 支持 | 支持 |
| Phonopy 有限超胞导出 | 支持 | 支持 | 支持 |
| 未折叠 FC2 晶格遍历 | 支持 | 不支持 | 不支持 |
| FC3/FC4 晶格导出 | 有这些阶数时支持 | 没有这些阶数 | 短程有这些阶数时支持 |

因此 ShengBTE FC3/FC4 的导出不受 FC2 长程叠加影响；TDEP 的总 FC2 晶格导出则不能仅凭 folded 数组完成。缺少所需来源时明确报错，不能自动省略长程部分。

若将来要求 Ewald 的有限范围晶格输出，应在 Ewald 侧另外定义截断范围、长程项及 onsite 的数学关系，再输出通用晶格张量数据；不得在 CompactForceConstants 中根据未知来源自动猜测平移，也不得宣称有限截断等于完整 Ewald。无限长程求和与有限晶格列表之间的差别无法通过新增数组标签消除。

### 实施顺序

1. 建立 CompactForceConstants 的短程转换、folded 数据输出、分块读取与 FC2 收缩。
2. 将 Ewald 初始化拆为原胞准备，目标超胞求值迁入显式 mapping 的两个方法；保持求和与修正算法不变。
3. 实现同类型相加并迁移格式 writer，删除被替代的展开和格式路径。
4. 更新 ForceDataset 扣除、有限差分及 NaCl 教程示例；验证同一 mapping 的力扣除和张量回加一致。
5. 覆盖不同超胞、非对角超胞、原子顺序、数值单位、未折叠信息缺失及不必要的 full 物化；文档与 docstring 同步描述新生命周期。

## 实施状态：上述接口修订已落地

当前生产入口为 `Ewald(cluster_space, ...)`、`ewald.forces(mapping, displacements)` 和 `ewald.force_constants(mapping)`。后者返回 `CompactForceConstants`，与短程模型转换结果同类型；力计算可通过该对象的 `harmonic_forces` 复用。旧 Ewald 的超胞绑定、compact_fc2 属性及 full_fc2 方法已移除。

统一张量实现位于 `force_constants/compact.py`，格式编码仍位于 `formats.py`。参数模型的 write 委托统一实现；原 `export.py` 中重复的折叠和全量展开路径已删除。短程和 folded 来源相加后逐块求值，格式阈值在相加后应用。晶格标签索引可以复用，但不缓存完整阶数的数值张量。

Phonopy/Phono3py 保持现有 full 文件布局，按多原子轴分块写出；Phonopy 文本不再收集完整文本列表。ShengBTE 和 TDEP 使用两遍晶格遍历完成计数和写出，不保留完整数值或文本列表。HDF5 的 p2s_map 使用 mapping 中零平移的 primitive anchor，不再使用输入原子顺序中的首次出现。原生 v5 HDF5 的参数模型存储保持不变。

当前 Ewald 输出仅具有目标超胞的 folded 信息。总 FC2 的 TDEP 晶格导出不支持；拥有短程 FC3/FC4 的组合仍支持相应 ShengBTE/TDEP 导出。没有实现无限长程的有限范围晶格列表或任意 q 的新求值器；不通过附加标签假装恢复丢失信息。

## 最终接口调整：绑定超胞，移除 TDEP

本节替代前面的显式 mapping 复算设计，前文保留为讨论记录。当前正式入口恢复为 `Ewald(mapping, ...)`；初始化一次性计算该超胞的完整长程 compact FC2。`ewald.forces(displacements)` 复用该结果，`ewald.force_constants()` 返回同一个通用 CompactForceConstants 对象。更换目标超胞时构造新的 Ewald，不在方法调用中切换超胞。

```python
ewald = Ewald(mapping, born_charges=born, dielectric=epsilon)
residual = data.subtract_forces(ewald.forces(data.displacements))
short_model = FitSystem(residual).solve()
short_model = short_model.enforce_asr().force_constants
total = CompactForceConstants(short_model, mapping) + ewald.force_constants()
total.write("force_constants.hdf5", format="phonopy", order=2, storage="hdf5")
```

TDEP writer、格式选项及其专属测试已删除。当前外部导出仅支持 Phonopy FC2、Phono3py FC3、ShengBTE FC3/FC4。原生参数化 HDF5 的 save/load 不变。

晶格信息限制保留在张量读取层：只有 folded 来源的阶数不能提供独立平移的 lattice_blocks。Ewald 只贡献 FC2，不影响短程 FC3/FC4 的 ShengBTE 导出；叠加后的总 FC2 可直接导出 Phonopy，不需要独立镜像标签。前文 TDEP 的约束仅是历史设计背景，不再代表现有导出功能。


## 模块归属调整与 NAC 后续研究

以上设计记录保留。当前数据入口已合并为 `mlfcs.dataset` 单文件，Ewald 已迁移至 `mlfcs.phonon.ewald`；此前提到的 `dataset/ewald.py` 是历史路径。通用张量与 IO 仍不依赖 Ewald。此次不改变数值算法，也未实现 NAC。现有声子接口缺口、Γ 极限和任意 q 长程接入范围详见 [NAC 调研方案](nac-plan.md)。
