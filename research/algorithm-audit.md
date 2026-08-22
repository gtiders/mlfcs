# MLFCS 数学与算法结构审查计划

## 1. 目标与边界

本计划审查 MLFCS 中是否存在“先枚举再过滤”或自行实现成熟数学算法的路径，并优先使用群作用、整数格理论、张量表示和标准数值线性代数减少重复计算。

本计划不改变以下范围：

- 一次拟合只使用一个确定的 reference supercell；
- canonical IFC 继续使用 primitive site 与 exact integer translation $R$；
- HNF 继续负责有限整数商格的 reduction、representative 和 cell index；
- Wick、Taylor、ASR、IFC 和 HDF5 的物理语义保持不变；
- 不引入多 supercell 联合拟合；
- 不以近似随机算法替代严格拟合。

## 2. 当前主要计算链

```text
primitive + reference supercell
        ↓
StructureRelation / HNF quotient / atom matching
        ↓
primitive exact-R interaction enumeration
        ↓
space-group and tensor-index orbit reduction
        ↓
single-supercell realization
        ↓
finite difference or Wick fitting
        ↓
constraints / streamed Gram / solver
        ↓
Wick-to-Taylor transform
        ↓
primitive exact-R sparse IFC
        ↓
target realization / IO / SSCHA / SCPH
```

## 3. 候选分类

### 3.1 Strong candidate

1. **Wick→Taylor intertwiner**

   当前实现为每个目标 orbit 堆叠全部 symmetry images，再对每个 source/target 阶数组合重复执行最小二乘。该过程本质上是两个对称参数空间之间的线性 intertwiner：

   $$
   T_{n\rightarrow n-2k}:V_n^G\rightarrow V_{n-2k}^G.
   $$

   应为每个目标 orbit 预计算一次 dual/left-inverse，并将它按 exact image key 分块。每个 contraction 随后直接通过对应 dual block 回代并累加，不再重复堆叠零块或调用 `lstsq`。

2. **Cartesian invariant 数值核标准化**

   当前实现自行使用 RREF 求 $C^TC$ 的零空间并选择独立 tensor components。应改为：

   - 对小型对称 Gram 使用 Hermitian eigensolver；
   - 用 LAPACK QR 秩判定选择独立 Cartesian rows，同时保持既有的高编号优先顺序；
   - 用线性方程求解完成 pivot normalization，不显式求逆。

3. **interaction orbit 的群作用构造**

   当前遍历全部空间群操作、全部锚点和全部 $n!$ 指标置换。本质上是有限群作用、orbit 和 stabilizer 问题。后续应研究生成元、Schreier tree、stabilizer generators 和 canonical augmentation。

4. **精确 reciprocal labels（已完成）**

   q 点及 $-q$ 的配对应使用 reciprocal quotient 中的整数群坐标，不应将浮点 q 坐标舍入为 hash key。

5. **SCPH/SSCHA fixed-point acceleration**

   当前线性 mixing 可扩展为带保护和重启的 Anderson/Pulay mixing。目标 fixed point 不变，只改变迭代路线。

6. **有限差分位移构型的群轨道约化（已推导，未实现）**

   中心差分的 $2^{n-1}$ tensor-product stencil 本身保留，但完整 signed displacement configurations 可以按 reference symmetry 分轨道，只计算代表构型并旋转恢复力。

### 3.2 Needs benchmark

- 将高阶 compatible-tail 递归识别为带重复标签的固定大小 clique enumeration，预计算 adjacency bitset，并使用交集式搜索或 orderly generation；
- 直接使用 symmetric tensor-power 结构构造 label-symmetric basis；
- constraint/identifiability 使用 structural rank 预筛，并在大 block 上评估稀疏 rank-revealing QR；
- 在有限平移群上使用 quotient-group FFT 批量构造 dynamical matrices；
- 使用 irreducible q mesh 并严格恢复 covariance tensor blocks；
- 对 dense reduced Gram 评估 block-Jacobi 或 pivoted-Cholesky preconditioner；
- 对有限差分外推的固定 Vandermonde matrix 只分解一次并处理多个右端项。

### 3.3 Already near-optimal

- SymPy HNF-backed `IntegerLatticeQuotient`；
- SciPy Hungarian structure matching；
- ASE Minkowski reduction/general MIC；
- 不显式构造 Kronecker matrix 的逐轴 tensor contraction；
- streamed design 与 BLAS dense SYRK；
- 必须显式产生全部条目的 IFC expansion 和格式稠密化。

### 3.4 No-go

- 用完整 SNF 替换 HNF atom mapping；
- 用 generic graph automorphism 编码已知的晶体空间群作用；
- Direct-Gram、全局 Wick feature correlation 或六阶 moment tensor；
- 需要反复重新执行 design kernel 的 matrix-free LSQR/LSMR；
- 默认使用 randomized sketching 改变严格最小二乘；
- 为当前低阶 Cartesian IFC 全面引入 spherical irreducible tensors。

## 4. 第一阶段：Wick→Taylor intertwiner

### 4.1 修改

对每个 target orbit 构造

$$
C_o=
\begin{bmatrix}
C_{o,1}\\
C_{o,2}\\
\vdots
\end{bmatrix},
$$

其中 $C_{o,i}$ 是第 $i$ 个 exact image 的参数列。通过 economy QR

$$
C_o=Q_oR_o
$$

预计算左逆

$$
L_o=R_o^{-1}Q_o^T,
\qquad
L_oC_o=I.
$$

将 $L_o$ 按 image row range 分成 $L_{o,i}$。对于收缩后的 image tensor $Y_{o,i}$，直接计算

$$
\Delta T_o=
\sum_i L_{o,i}Y_{o,i}.
$$

这与原来的联合最小二乘严格等价，但 target orbit 的分解只执行一次，也不再构造跨 source parameter 的大零块矩阵。

### 4.2 验证

- 保留一个仅供测试使用的旧联合 `lstsq` reference；
- 比较 FC4→FC2、FC5→FC3 以及多 source orbit 聚合；
- 比较完整 transform、Taylor parameters、约束残差、预测力和 sparse IFC；
- 记录 transform 构造时间和峰值临时数组。

## 5. 第一阶段：tensor invariant 数值核

### 5.1 修改

当前 stabilizer constraint 的 Gram 为

$$
G=\sum_g (A_g-I)^T(A_g-I).
$$

它是小型对称半正定矩阵。使用 `scipy.linalg.eigh()` 求其零本征空间，避免对 $G$ 手写 RREF。阈值统一相对于最大谱尺度定义。

对得到的 invariant basis $B$，按照既有的反向 Cartesian 顺序逐行加入候选集合，并使用 LAPACK QR 判断秩是否增加。这样既不维护本地消元算法，也保持有限差分 displacement keys 的历史定义。最后解

$$
B_P^TX^T=B^T
$$

获得满足 $B_P=I$ 的 normalized basis，不使用显式矩阵逆。

### 5.2 验证

- FC2、FC3、FC4 和小型 FC5 的 invariant dimension 保持一致；
- 每个 stabilizer action 满足 $\|A_gB-B\|$ 在容差内；
- pivot submatrix 满秩且 normalization 后为单位阵；
- reconstruction、有限差分、拟合和约束测试通过；
- Si、SnSe 的 orbit 数、参数数、预测力、RMSE 和最终 Taylor IFC 在浮点容差内一致。

## 6. 后续阶段

第一阶段验证通过后，按以下顺序继续：

1. 谐偶极长程力扣除的独立数值核与 KCl 回归；
2. signed displacement symmetry 作为默认关闭的独立有限差分调度功能；
3. compatible-tail clique/canonical augmentation benchmark；
4. SCPH/SSCHA Anderson mixing prototype；
5. FC6+ generator-based orbit 内存原型；
6. 根据实际 profile 决定 sparse QR、irreducible q mesh、quotient FFT 和 Gram preconditioner。

每个阶段均独立提交；未达到数值等价或实际收益门槛的候选完全撤销，不保留运行时双分支。

space-group generator prototype 已完成：FC2–FC4 的数学结果严格等价但整体替换为 No-Go；Si FC5 仍无收益，Si FC6 的纯 key traversal 则由 16.80 s 降至 7.70 s，因此 FC6+ 标记为有价值的独立候选。它必须先完成 tensor invariant prototype，当前不进入生产或拟合路径。详细结论见 `research/orbit_generators/HIGH_ORDER_GENERATORS.md`。

exact reciprocal labels 和 q 点回归也已完成：有限群整数 label 直接给出 $q/-q$ 配对，不再使用浮点 q 坐标舍入作为 key。对角和非对角 HNF quotient、SSCHA 配对及 SCPH 单/多 worker 回归已通过。详细结果见 `research/qpoints/REGRESSION.md`。

## 7. 第一阶段执行记录

已完成：

- target-orbit QR 只分解一次，left inverse 按 exact image key 缓存为 dual blocks；
- 每个 source contraction 直接执行 dual-block contraction，不再构造跨 source parameter 的零填充联合矩阵；
- invariant null space 改用对称 `eigh`；
- independent Cartesian rows 改用保持历史顺序的 QR rank test；
- pivot normalization 改用 `solve`，删除显式矩阵逆。

验证结果：

- 随机多 image、多右端项的 dual-block 回代与联合 `lstsq` 在 $10^{-13}$ 量级一致；
- Si 小型 FC2+FC3+FC4 transform 与修改前最大差异为 $4.34\times10^{-19}$；
- 同一小型基准的 transform 构造时间由约 $0.211$ s 降到约 $0.046$ s；
- Si 实际 cutoff 下 FC2、FC3、FC4 分别保持 $4$、$10$、$18$ 个 orbit，pivot indices 逐项一致，normalized basis 最大差异分别为 $8.88\times10^{-16}$、$8.88\times10^{-16}$ 和 $1.76\times10^{-15}$；
- SnSe 实际 cutoff 下 FC2、FC3、FC4 分别保持 $55$、$204$、$99$ 个 orbit 和 $400$、$4354$、$3818$ 个参数；pivot indices 逐项一致，normalized basis 最大差异不超过 $9.99\times10^{-16}$；
- FC2、FC3、FC4 invariant basis 逐 orbit 通过 stabilizer invariance 和 pivot normalization 检查；
- 完整测试集通过；
- 初次使用自由 pivoted QR 曾改变有限差分 displacement keys，并使 Morse FC4 RMS 超过既有阈值。该方案已撤销，最终实现明确保留原来的高编号 Cartesian pivot 优先语义。

### 7.1 真实拟合回归

- Si 100 帧 FC2+FC3+FC4 无缓存重算完成；训练 RMSE 为 $3.9540507309\times10^{-2}$ eV/Å。旧、新 sparse keys 与 translations 完全一致，FC2、FC3、FC4 tensor 最大差异分别为 $3.91\times10^{-14}$、$1.31\times10^{-13}$ 和 $1.28\times10^{-11}$。$4\times4\times4$ phonopy mesh 的最大频率差为 $6.93\times10^{-14}$ THz。
- SnSe 201 帧数据按原脚本拆成 181 个训练帧和 20 个验证帧，无缓存重算完成；训练/验证 RMSE 分别为 $1.6255268400\times10^{-2}$ 和 $1.8025135486\times10^{-2}$ eV/Å。旧、新 sparse keys 与 translations 完全一致，FC2、FC3、FC4 tensor 最大差异分别为 $7.19\times10^{-8}$、$3.65\times10^{-7}$ 和 $7.62\times10^{-6}$；相对于各阶最大 tensor，其量级约为 $10^{-8}$ 至 $2\times10^{-7}$。$4\times4\times4$ phonopy mesh 的最大频率差为 $5.74\times10^{-8}$ THz。
- Ba8Ga16Ge30 的 300 K、101 帧 FC2+FC3 有效 IFC 无缓存重算完成；训练 RMSE 为 $2.6403982428\times10^{-2}$ eV/Å，历史值为 $2.64039824277\times10^{-2}$ eV/Å。使用同一 Gram 和参数化分别执行旧联合 `lstsq` 与新 dual-block intertwiner 后，FC2、FC3 sparse tensors 逐元素完全一致。
- 三个案例的谐波声子图均已重新生成。Si 无虚频；SnSe 保留原拟合中 Γ 附近约 $-0.4$ THz 的软支；Ba8Ga16Ge30 的 300 K 有效 FC2 图无明显虚频。
- Ba8Ga16Ge30 顶层静态 FC2+FC3+FC4 脚本当前不能作为回归：训练快照是 54 原子晶胞，而脚本传入 432 原子 reference，拟合在训练数据验证阶段明确拒绝。该输入组织问题与本阶段算法修改无关，未通过静默复制原子或更换参数绕过。

### 7.2 q 点与非谐案例回归

- 对角、非对角剪切矩阵的 HNF reciprocal quotient 测试通过；新旧方法的 q 点集合一致，canonical ordering 允许改变。
- SSCHA 的 $q/-q$ 配对和 SCPH 单/多 worker covariance 回归通过。
- K4As4Pt2 SCPH 使用 300/600/900 K、mixing 0.5、200 步和 4 workers 重跑并绘图；三个温度均停在约 $10^{-6}$ THz 的变化平台，未达到 $10^{-10}$ THz 严格阈值，但结果无 NaN，最低频率仅为约 $-10^{-4}$ THz 的零模数值尺度。
- K4As4Pt2 SSCHA 使用 300 K、100 快照和 5 次更新重跑并绘图；12 个 q 点和 357 个模式均正常采样。
- 详细记录见 `research/qpoints/REGRESSION.md`。
