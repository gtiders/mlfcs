# Numba 迁移前数学审查

## A. 参考与真实调用链

本报告先于生产代码修改建立。Python 原点为 HEAD `c930254`，核心算法历史为
`a08c9ee`（`git show a08c9ee:src/mlfcs/cluster_space/invariants.py`）；Rust 工作保存为
stash `a3f84bdf803338e9364e093f2222cd0c918f5c3f`，说明为
`rust integer-safe backend reference before numba migration`。
Rust 源码位于该 stash 的第三父提交，Python 接口修改在 stash 主树。
`/tmp/mlfcs-rust-reference` 导出二者并保存现有 CPython 3.14 扩展，
`/tmp/mlfcs-python-reference` 导出原 HEAD，供隔离进程差分。
Git 不会 stash ignored 编译产物；这些已另外复制，源码在 stash 中完整保存。
`alamode/` 是独立未跟踪目录，未纳入迁移。

调用链：

1. `PrimitiveCell.__post_init__` 包装 fractional positions 到 $[0,1)$，用
   `spglib.find_primitive` 确认 motif 已是 primitive。
2. `PrimitiveSymmetry.from_primitive` → `spglib.get_symmetry_dataset` →
   rotations、translations、site_permutations、site_shifts、cartesian_rotations。
3. `build_cluster_space` → `_build_order` → `iter_candidates` → `_neighbors`。
   Python 原版用 ASE neighbor list；Rust `neighbors` 使用 inverse-cell 包围盒，
   `enumerate_candidates::extend` 与 Python `extend(location)` 同为可重复非降序枚举。
4. `_orbit_actions` → Numba `transform_cluster`；Rust 对应 `build_orbits` →
   `transform_cluster`。首个未覆盖 candidate 保持代表元，群操作和轴排列生成 images/stabilizers。
5. `invariant_basis` → `label_symmetric_basis` → `apply_lattice_action` →
   `certified_pivots` → `saturated_kernel` → label basis 扩展。
   Rust 对应 `label_basis` → `rotate_tensor` → `independent_rows` → `integer_kernel`。
6. `_tensor_frame` 的 $F_p=(A^T)^{\otimes p}$ 将 exact lattice basis 转成 Cartesian，
   `component_parameterization` 用 SciPy QR 和行选取构造可观测参数。
7. `ClusterMap.build` 周期折叠，`rank_info` 将 exact_lattice_basis 经群作用后
   对相同 atom tuple 累加，求精确 folded rank 和 aliases。
8. `ForceDesign._compile_order` → Numba `_accumulate` → fitting；
   `ForceConstants`/lattice expansion → force constant representations 与 Taylor calculator。

入口已知道 $N,p,S,A,r_c,R,H$；不需要假设任意未知群或整数矩阵。
equivalent_atoms 不用于当前枚举；orbit 和 parameter dimension 是后续结果，
不能提前假设对称性必然降低数量。

## B. 严格上界

令 $L=2^{63}-1$，$R=\max|\mathrm{rotation}|$，$H=\max|\mathrm{site\_shift}|$。
实际阶数集合为 $\mathcal{P}$，$P=\max\mathcal{P}$。
primitive cell 的 ASE 行矩阵为 $A$。设
$\rho_j=r_c\|(A^{-1})_{:j}\|_2$。
若 displacement 在 cutoff 内，Cauchy–Schwarz 给出
$|t_j+\delta_j|<\rho_j$，其中 $\delta_j=x_{site,j}-x_{anchor,j}\in(-1,1)$。

Rust 实际搜索扩大到 $b_j=\rho_j+1$：

$$
\ell_j=\lceil-b_j-\delta_j\rceil,\quad
u_j=\lfloor b_j-\delta_j\rfloor,\quad
n_j\le\lfloor2b_j\rfloor+1,\quad K_p=\prod_j n_j.
$$

Numba 实施将在 Python 层把输入 float 的精确二进制有理数用 `Fraction`
表示，并精确求逆。平方根使用有理数向上界：对于 $q=a/b\ge0$，
$(\operatorname{isqrt}(ab)+1)/b>\sqrt q$。这样避免把普通浮点求逆的舍入
当成整数证明。使用该向上界构造同一扩大搜索盒；搜索本身采用 float64
距离，沿用原模型 cutoff 语义。该证明针对搜索盒内所有整数循环，
不声称 float64 cutoff 是实数几何的精确谓词。

| 量 | 来源 | 严格上界 | int64 条件 |
| --- | --- | --- | --- |
| $M_p$ | 每 anchor neighbor labels | $K_pN$；生成后可取真实最大值再收紧 | $K_pN\le L$ |
| $T_p$ | 搜索 translation 分量 | $\max_j\lceil b_j+1\rceil$ | $T_p+1\le L$（循环端点） |
| $C_p$ | 非降序可重复尾部 | $N\binom{M_p+p-2}{p-1}$ | $C_p\le L$ |
| $O_p$ | 未覆盖 candidate 生成 orbit | $C_p$ | 同上 |
| $D_p$ | Cartesian tensor rows | $3^p$ | $D_p\le L$ |
| $Q_p$ | 轴排列 | $p!$ | $Q_p\le L$ |
| $I_p$ | 所有 orbit images / actions | $C_pSQ_p$ | $I_p\le L$ |
| $d_p,B_p$ | orbit dimension / 参数总数 | $d_p\le D_p$，$B_p\le D_pC_p$ | $\sum B_p\le L$ |
| 群作用标签 | $Rt+h$，再 re-anchor | $2(3RT_p+H)$ | 此量 $\le L$ |
| seed tensor action | 0/1 label basis 逐轴 contraction | $U_p=\max(1,3R)^p$ | $U_p+1\le L$ |
| constraints | $T(seed)-seed$ | 每项绝对值 $\le U_p+1$ | 同上 |
| constraint buffer | 最多 $SQ_p$ 个 stabilizers | $SQ_pD_p^2$ 个元素 | 元素数与 bytes 都 $\le L$ |
| transformed labels | 群作用缓冲 | $4pSQ_p$ 个元素 | 同上 |
| 全部 label 存储 | 所有 images | $4pI_p$ 个元素 | 同上 |
| 全部 image bases | 每 image $D_p d_p$ | $I_pD_p^2$ 个元素 | 同上 |
| folded rows / alias count | 按 tuple 汇总 | 行数 $\le I_pD_p$；aliases $\le I_p$ | 同上 |
| dense folded matrix | 行数 × 参数列 | $I_pD_pB_p$ 个元素 | 同上，或坚持 sparse/bigint orchestration |
| exact kernel coefficients | unimodular 消元 | 未获得适用于当前算法的实用 int64 上界 | Python bigint |
| folded coefficients | kernel rotation + alias 累加 | 含实际 kernel 最大值；不能由 $B_p$ 推出 | Python bigint |

组合式来自 `extend(location)` / Rust `start=i`，尾部长度 $p-1$，
pairwise cutoff 和 body_order 只会删除候选。
因此

$$
B_{total}\le\sum_{p\in\mathcal P}3^pN\binom{K_pN+p-2}{p-1}.
$$

参数 start/stop/offset、orbit index 可由此约束；image count、flattened
index、tensor coefficients 不能仅由此约束。尤其 $S,p!$ 和 $R,H,T_p$
必须出现在证明里。不硬编码 $K=27$ 或 $S=48$。

## C. Safe domain 与实际限制

采用 $F$ 为上表所有 fixed-width 数量、运算绝对值，以及每个待分配
int64/float64 数组元素数乘 8 的最大值，并要求 $F\le L$。
多阶累计 offsets 还要求 $\sum B_p,\sum I_p,\sum I_pD_p^2\le L$。
这是充分条件，保守拒绝不代表任务必然溢出。
shape 必须同时满足 `np.intp` 平台范围；不假设平台必然为 64 位。
Rust 64 位 usize 的 $2^{64}-1$ 更宽，以上条件也覆盖其索引范围。

supercell 与 snapshots 在线性代数入口才已知，不能由 primitive 输入证明：
设 $V=|\det(supercell\ matrix)|$，$A_s=NV$，$J$ 为线程数，$E$ 为样本数，
需另验证 $3A_sB_{total}$、$I_ppV$、$J3A_sD_p$、$E3A_sB_{total}$
及 bytes/index 范围。周期 quotient 的 adjugate/determinant 保留 bigint，
避免 supercell matrix 未限制时偷偷使用 int64 乘积。

仅看 parameter 条件，固定 $K$、单阶 $p=3$ 时
$B_3=27N(KN)(KN+1)/2$。
例如 $K=27,N=2$ 给出 $B_3\le80190$；$N=200$ 给出
$B_3\le78746580000$（单 dense basis 超过 629 GB）。
该模型的参数 overflow 阈值可用 bigint 二分精确求出；这是条件示例，
不是实际任务的 $K$。完整 $F$ 的阈值更低，因为全 image basis
乘 $SQ_pD_p$，dense folding 又乘 $B_p$。
不把“内存先用完”当证明；内存/时间与数学阈值分别记录。

## D. i128 调查

Rust 文件唯一 i128 表达式在 `linear_sub`：
`i128::from(a) - i128::from(q) * i128::from(b)`，随后 `i64::try_from`。
调用图：`build_model → build_orbits → invariant_basis → independent_rows →
integer_kernel → linear_sub`；另有 `invariant_basis → integer_kernel` 和
`folded_rank → rank_rows → independent_rows → integer_kernel`。
因此它在精确整数核的 Euclidean 行/列消元及 unimodular 变换追踪中，
位于 Cartesian conversion 之前，也可在 folded rank 阶段执行。
不是单纯 Cartesian conversion 的辅助保险。
扩大乘积避免 $q b$ 溢出，即使最终 $a-qb$ 可落回 i64；
Rust 仍拒绝最终 coefficient 超范围。
Python 原版用 SymPy `smith_normal_decomp`，内部 bigint，
没有要求 128 bit，也没有以 Rust fixed-width 类型定义数学域。
实际材料系数范围需在差分脚本运行中报告，不能从小测试推及所有晶胞。

## E. 精确核决策与坐标表示

选择：**Python bigint 精确 saturated kernel + Numba modular rank**。
当前 integer output 继续用于 exact folding/rank，并公开为 exact_lattice_basis。
folded rank 只需要 spanning rational basis，理论上并不必需 saturation；
现有 `saturated_kernel` 的公开契约与测试要求 lattice saturation，故保留。

备选比较：

| 路线 | 结论 |
| --- | --- |
| Euclidean unchecked int64 | constraint 小不保证 unimodular 变换小；拒绝无证明迁移 |
| Euclidean 局部 checked int64 | 可行但会拒绝原 Python 可接受的 bigint 数据，不作为正式路径 |
| Bareiss | minors/Hadamard 限制最终值，但乘法中间值仍可宽于 int64；不能自动解决 |
| modular kernel + CRT/reconstruction | residues 的乘积安全；重建与 saturation 仍须 bigint 和精确验证，实施复杂 |
| modular rank + floating QR/SVD | 可用于 Cartesian 子空间，不能单独替代 exact folded-rank 数据 |
| signed union-find | 仅适用于实测 signed permutation 子类；hcp/sheared lattice 不满足一般条件 |
| bigint Smith kernel | 现成原 Python reference，保留 exact lattice 契约，不依赖 int128 |

`certified_pivots` 已实现 distinct primes product + Hadamard bound 证明 exact rank。
单个 prime 的 rank 只是 lower bound；不是 exact-rank 证书。
modulus $2\le q<2^{31}$ 时 normalized residues 满足
$0\le x<q$，$xy\le(q-1)^2<2^{62}$，
$x-yz$ 也位于 int64。公开 modular 入口必须验证 prime 和范围。

spglib rotations 属于输入 cell 的 fractional lattice 坐标，非 Cartesian，
也不自动属于 standardized cell。
见 [spglib 坐标定义](https://spglib.readthedocs.io/en/stable/definition.html)。
本项目 row convention 为 $x'=xR^T+t$，Cartesian row action
$C=A^{-1}R^TA$，tensor column frame $A^T$ 与 $R$ 交织。
整数 unimodular shear 会使 lattice rotations 的元素很大，
即使 Cartesian rotation 正交；不能假设 $R=1$ 或 signed permutation。
QR/LAPACK 留在 Python/SciPy 层；
[SymPy DomainMatrix](https://docs.sympy.org/latest/modules/polys/domainmatrix.html)
提供整数域精确运算，不将 bigint 转 float 来模拟 exact kernel。

## F. 迁移结构与验收

`core/algebra/domain.py`：Python bigint bounds、shape/byte validation。
`cluster_space/_candidate_kernel.py`：Numba neighbor search + streaming DFS。
`cluster_space/_orbit_kernel.py`：现有 Numba 群作用，入口统一 proof。
`cluster_space/_tensor_kernel.py`：Numba int64 逐轴 contraction；
bigint basis 使用 NumPy object 运算。
`core/algebra/exact.py`：Numba modular elimination，Python bigint Smith kernel。
`supercell/map.py`：Python bigint quotient 和精确 folding，modular rank。
已有 fitting/Taylor Numba 数值循环继续使用，增加输入 shape 安全边界。
Python orchestration 拥有模型、集合去重、spglib、SciPy；生产路径不含 Rust。

先恢复原 Python 测试；实现域验证，再迁移 neighbor/candidate/tensor，
建立 HEAD Python / stash Rust / 当前 Numba 隔离差分。
对 cluster labels、orbit/image/mapping、参数数、rank/alias 完全比较；
任意 exact kernel basis 比较精确 annihilation、维数及 rational span，
不能要求两个不同合法 saturated bases 的列元素相等。
Cartesian 基比较正交化 projector、residual 和 nullity，并验证最终模型。
保留 Rust stash，不 pop，不删源码参考；编译产物移至临时参考目录。
benchmark 报告 cold compile 和 warm execution，不能把 JIT 编译时间算作稳态。

实施/验证结果在代码迁移后追加；本报告此时未声称迁移完成。

# G. 无 i128 的 exact integer kernel 算法研究

本章追加于 A–F 之后。前六节原文逐字保留；追加前 SHA-256 为
`b506d7008f410ff1bccdc623627e1e930ceed93ec3fbf46badc7111c1a957d0c`。
Rust 参考仍是 stash `a3f84bdf803338e9364e093f2222cd0c918f5c3f`，没有 pop。
本节是算法研究，不改生产代码，也不把候选算法描述成已经实施。

## G.1 当前 i128 的根源：A 和 V 都会长，V 的追踪不可省略到当前方法里

令当前消元矩阵为 $A_k$，完整列变换为 $V_k$。Rust 代码每次做列操作
$C_j\leftarrow C_j-qC_k$，同时做
$A_k[:,j]\leftarrow A_k[:,j]-qA_k[:,k]$ 和
$V_k[:,j]\leftarrow V_k[:,j]-qV_k[:,k]$。最后取 $V_k$ 中消元后为零的列，
作为原坐标中的整数 kernel generator。因而 `linear_sub` 同时用于 A 与 V；
不是只有变换矩阵在乘法前扩宽。

`i128` 的直接原因是通用的 `a - q*b` 实现必须先形成乘积；两个 i64 操作数
的乘积可超出 i64，即使抵消后的差仍处于 i64。这个情况在整数算术上确实存在，
但本次运行的真实 constraint 样本没有出现“乘积超 i64、结果回到 i64”的事件：
相反，样本中的主要问题是被存储的 A、V 与结果本身在普通 Euclidean 商减后变大。
因此单独换一种乘法写法，不能解决实测 coefficient growth。

profiling 采用与 Rust 一致的首个最小绝对值 pivot、截断 toward-zero 商、行/列交换、
先 row elimination 再 column elimination，并分别记录 A 与 V；小整数计数器用
Python 整数跟踪，以免 profiler 自身溢出。每个案例最多运行 8 秒或 2,000,000 次更新；
165/165 个样本全部完成，最后验证了原始约束矩阵乘 kernel 等于零。
统计中的 V 更新 `q*b` 上界是发生过的最大临时乘积绝对值；不是只在乘积大于 int64
时才计数。全样本没有出现 `|q*b| > int64_max` 且 `|a-q*b| <= int64_max` 的抵消事件。
这表明 `i128` 是 Rust 实现为任意当前 int64 操作数保留可检测的乘积安全余量，
而真实输入更显著的风险是消元结果本身增长。

两个反例量化了这种增长：

| 样本 | 初始 max $|A|$ | max 存储 $|A|$ | max 存储 $|V|$ | max $|q|$ | max $|qV|$ | kernel max | rank / nullity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Si diamond，anchor onsite FC5 | 30 | $1.4373\times10^{39}$（131 bit） | $8.2791\times10^{50}$（170 bit） | $3.2833\times10^{19}$（65 bit） | $8.2791\times10^{50}$ | 15 | 20 / 1 |
| Si，等价 shear $U_{12}=31$，FC4 representative 6 | $1.0952\times10^{12}$ | $3.1517\times10^{19}$（65 bit） | $3.3516\times10^{19}$（65 bit） | $2.1072\times10^{15}$ | $3.6789\times10^{19}$ | $1.8882\times10^8$ | 66 / 15 |

FC5 示例有 21 列、rank 20，kernel 只有一列；当前 Euclidean 算法却让中间变换 V
长到 170 bit，且 A 长到 131 bit。FC4 shear 示例有 81 列、486 行原始 constraint、
rank 66/nullity 15；它的 lattice rotations 最大元素为 1023，来自等价的整数基变换。
这说明 A 也会增长，且这两个有限样本中不能把现象归因于 V 一项。
Rust i64 版本会在这些后续更新到达 i64 边界时返回 OverflowError；profiling 的大整数
只用于揭示同一更新序列继续执行时的增长，不表示 Rust 实际接受了超宽值。

反过来，固定已有测试与材料结构、cutoff 4 Å，从候选枚举中按首个未覆盖 cluster 取样：
Si 21、Mg 19、单 motif 七个晶系共 51、graphene 8、MoS2 8、K4As4Pt2 12、
Ba8Ga16Ge30 8 个 representative；另有 shear 7 和 shear 31 的 Si 各 17 个、
Si/Mg onsite FC5/FC6 各 2 个。共 165 个精确实例。FCC diamond / hcp 的 $S$
分别为 48 / 24；材料数据中的 $S$ 为 3–24。未声称枚举完整材料模型的每一个 orbit。
对每个 instance 同时统计 shape、row/column nnz、重复行、初始系数、rank/nullity、
rotation 最大值、消元和 kernel 系数；中间明细及可复现临时脚本的 SHA-256 分别为
`ee1a6825ab8b2808a3f8d312b5142099c6df56daa38013015e8bdc8cffe557da`（profile JSON）与
`9b8eeafae5e202a37e367bdd5c1b35a0a7187c233c13f948a3b1d3bad4d1079`（profiler）。
JSON 当前在 `/tmp/mlfcs-kernel-profile.json`，脚本在 `/tmp/mlfcs-kernel-profile.py`；
前一个 hash 应以重新运行 `sha256sum` 为准，不作为报告基线。

在 165 个实例里，84 个约束矩阵所有行可约化成一个非零项或两个等幅非零项；
它们包括 0/±1 的 signed graph 约束。剩下的 81 个实例有真正多变量小整数关系，
shear 样本则是稠密且系数很大的等价表示。示例范围：

| family | instance count | $S$ | 初始 $|A|$ max | 消元 $|A|$ max | 消元 $|V|$ max | $|qV|$ max | final kernel max | signed graph cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| triclinic–cubic one-site | 51 | 2–48 | 1–2 | 1–6 | 1–6 | 1–12 | 1–2 | 47/51 |
| Si diamond | 21 | 48 | 1–12 | 1–30 | 1–56 | 1–60 | 1–24 | 4/21 |
| Mg hcp | 19 | 24 | 1–6 | 1–9 | 1–12 | 1–16 | 1–6 | 7/19 |
| graphene / MoS2 | 16 | 12–24 | 2 | 2 | 2 | 2 | 2 | 7/16 |
| K4As4Pt2 / Ba8Ga16Ge30 | 20 | 3–8 | 1–2 | 1–2 | 1 | 1 | 1 | 20/20 |
| Si shear 7 | 17 | 48 | 15,752,961 | $2.922\times10^{11}$ | $3.678\times10^{11}$ | $5.270\times10^{11}$ | 148,862 | 0/17 |
| Si shear 31 | 17 | 48 | $1.095\times10^{12}$ | $3.152\times10^{19}$ | $3.352\times10^{19}$ | $3.679\times10^{19}$ | 188,815,358 | 0/17 |
| Si onsite FC5/FC6 | 2 | 48 | 20–90 | $1.927\times10^{39}$ | $8.279\times10^{50}$ | $8.279\times10^{50}$ | 15–239,694 | 0/2 |
| Mg onsite FC5/FC6 | 2 | 24 | 20 | 20 | 28 | 48 | 6–10 | 0/2 |

`label_symmetric_basis` 确实先把重复 lattice site 的 tensor components 等同化；
Rust `label_basis` 是同一件事。它已缩减同类 Cartesian 分量，却不会使一般空间群作用
变成坐标置换。上表 row nnz 最大为 81，column nnz 最大值在复制轴排列的原始矩阵中
达到 4,056,480；unique-row 数远小于原始行数。按重复轴置换压缩行后，FC5/6 Si onsite
分别只有 208/331 个 unique rows，原始约束矩阵 shape 分别是 699,840×21 与
12,597,120×28。只保留独立的第一份 stabilizer action 可大幅去掉重复存储，不能据此
假定所有剩余行是 incidence rows。

这些数据是带明确采样规则的证据，不能证明全部允许输入的消元次数或 coefficient 上界。
特别是 high-order onsite constraint 中对所有 $p!$ 轴排列重复枚举行，是纯实现冗余；
因 seed 已强制同位点分量对称，可先构造实际不同的稳定作用。

## G.2 矩阵是什么、saturation 到底要求在哪里

对代表 cluster 的 stabilizer action $T_g$，令 $E_p$ 为 `label_symmetric_basis`。
Rust 生成的约束块是

$$
A_g=(T_g-I)E_p,
$$

矩阵列变量 $x$ 是 seed coordinates；精确 invariant tensors 是
$E_px$ 且 $A_gx=0$。`independent_rows` 先用一个大素数选择独立行，然后用
`integer_kernel` 验证所选行的 kernel 也消灭全部原行。Python 使用
`certified_pivots` 的多素数/Hadamard 证书选 row span，再调用 `saturated_kernel`。
两者求出的线性空间是 rational kernel；返回的 integer columns 是它与
$\mathbb Z^m$ 的满秩 lattice basis，即
$\ker_{\mathbb Q}(A)\cap\mathbb Z^m$，也就是饱和的整数 kernel。**精确核的饱和性
是返回值契约，不能只返回几个 annihilated vectors。**

但是 downstream 使用可分开看：

- `component_parameterization` 将 integer columns 转 float，再左乘 $A_{cell}^{\otimes p}$、
  QR，并选 Cartesian observation rows。它只使用 rational subspace；任何有限指数子格
  的列也张成同一个 float subspace。参数化不使用列的 integer-lattice index。
- `ClusterMap.rank_info` 把 basis 经过 image action 并按相同 folded atom tuple 累加，
  最后只询问 folded coefficient matrix 的 rank。Rust 的 `folded_rank` 同样返回
  rank 和 aliases。对 rank 来说只需要 rational span。Python 最终调用 SymPy sparse rank；
  Rust 通过 `rank_rows → independent_rows → integer_kernel` 给出精确行秩证书。
- 因而当前**最终物理计算没有读取 primitive invariant lattice 的 saturation index**；
  saturation 被 API 命名/数据类型契约要求，也提供一个 primitive 整数表示，但不是
  fitting 参数坐标的额外物理条件。若改为单独保存 canonical rational span 并明确改 API，
  可合理放宽 saturated lattice basis；当前迁移须先作这个语义决定。
- 对 folded rank，甚至 kernel basis 都不是后续所需：只要一个精确 rank 证明。
  若继续保持 basis 型中间对象，则一个 rational basis 足够，不要求饱和。

Rust `integer_kernel` 显式构造 $V\in GL_m(\mathbb Z)$，靠 unimodular 列操作保持整数
lattice，再取消元矩阵的零列。因此它返回饱和 kernel。Python `saturated_kernel` 使用
SymPy `smith_normal_decomp`，取 Smith 右变换中零不变量列；大整数与 saturation 均由
SymPy 保证。仅对返回列验证 $AK=0$、列数等于 nullity，**本身不足以证明 saturation**；
必须额外验证 lattice index 为 1，或使用 construction certificate。

## G.3 结构判断：图法有真子类，群平均是 rank 证书，非一般饱和 basis

### Signed union-find

若某 constraint row 经除去公因数后仅为 $x_i=0$、$x_i-x_j=0$ 或
$x_i+x_j=0$，可用 signed weighted union-find 记录
$x_i=s_i x_{root}$，$s_i\in\{\pm1\}$。一个 component 若有一致的符号关系，
primitive generator 就是 entries $\pm1$ 的 component indicator；若 cycle 要求
$x_r=-x_r$，则整数域无 2-torsion，root 必为零。这种 kernel 直接 primitive，
不做 elimination，Numba 也易实现。

适用检查必须逐行验证 `nnz <= 2` 且两个系数绝对值相同；重标号/符号归一后才能 union。
只对该分支证明 saturated。84/165 取样实例满足这个条件，其余不能强行塞入。
一般小整数三项关系，如 $x_1+x_2-x_3=0$，需 weighted linear relation，链式权重
可以按指数增长；普通 union-find 不适用。矩阵也不是普遍 TU/network matrix：三行
$\begin{bmatrix}1&1&0\\0&1&1\\1&0&1\end{bmatrix}$ 是 signed two-entry
rows，但行列式为 2。它的非定向奇环已是非 TU 反例；若出现 $2x=0$ 一行，该行单独
也提供非 TU minor。带非单位权重的 sheared matrix 更不是图 incidence matrix。

### 有限群作用 / Reynolds 平均

令稳定作用像群为 $G$，$g=|G|$，$W=\sum_{h\in G}T_h$。在特征零中
$P=W/g$ 是投影，$P^2=P$，image 正好是共同固定空间；这是有限群的 Reynolds operator
事实。见 [Magma finite group invariants](https://docs.magma.maths.org/html/CommutativeAlgebra/InvariantTheory/invariants.html)
以及 finite group 平均的定义。

$W$ 的每一列是整数 invariant generator，但这些列通常不生成 saturated lattice：
$G=\{1,-1\}$ 对 permutation module $\mathbb Z^2$ 作交换时，$W$
的列是 $(1,1)$、$(1,1)$，其生成 lattice 恰好 primitive；而将两个等价表示经非 unimodular
坐标基混合后，projector columns 可生成固定线的真有限指数子格。一般而言
$g\mathbb Z^m\subseteq W\mathbb Z^m\subseteq\ker_{\mathbb Z}(A)$，所以商的指数素因子
只能来自 $g$，但这个 inclusion 只说明 saturation index 可由 $g$ 消去，不能当成
$W$ 列已经饱和，也不能界定每个 primitive vector 的坐标只靠 $g$。具体地，整数 involution
$T=\begin{bmatrix}1&0\2h&-1\end{bmatrix}$ 的有限群为 $\{I,T\}$，于是
$W=I+T=\begin{bmatrix}2&0\2h&0\end{bmatrix}$。其像的饱和格由 $(1,h)^T$ 生成，
而 $W$ 的非零列是 $(2,2h)^T$，指数为 2；取 $h$ 很大，群阶仍为 2，primitive
invariant 的坐标仍可很大。这说明群阶限制 saturation index 的质因子，并不单独限制
saturated basis entries。

对本项目，可把 $W$ 用作 integer-free basis 的候选及 exact nullity 证书：先删除完全相同的
`tensor action`（同一作用的重复轴排列只给同一个 $T$），累加有界 int64 $W$；
检查 $AW=0$ 和 $W^2=gW$ 的 exact modular residues；用足够多模数按
$|AW|,|W^2-gW|$ 上界证明残差为零；再由整数 trace
$\operatorname{rank}(W)=\operatorname{tr}(W)/g$ 得 image dimension。
因为幂等矩阵在特征零下 trace 等于 rank，且 image 已被验证为 $A$ 的 kernel，
可得 exact rank/nullity，而无须整数消元。算术入口还要证明 $g\max|T|$、
$g\max|W|$、residual modular dot products 和 trace sum 均在既定界内。
若用户给自定义而非 spglib 产生的 symmetry，必须验证这些 action 构成有限群表示，
或直接验证上述 projector identities；不能信任仅满足 shape/permutation 的数组。

在 profiling 的 compressed repeated-label action 中，最高 $|W|$ 是 Si FC5 的 180、
Si FC6 的 540，shear31 FC4 为 $2.1904\times10^{12}$；都在 int64 内。
这只是实测，不替代入口 bound。作用像群 $G$ 的元素要去重，操作列表重复时须一致加权，
否则 $W^2=gW$ 未必成立。

Reynolds 平均不直接给饱和 lattice basis。可先选 $d$ 个 independent columns，再求它们生成的
子格 saturation；index 只含 $g$ 的质因子。对各 $q\mid g$ 做 $q$-saturation 时，
可在模 $q$ 下找出使 $Bz\equiv0\pmod q$ 的新增方向，再加入 $(Bz)/q$ 并重归一。
若一直只保留模 $q$ 的 basis 坐标，乘加可 reduced modulo $q$；但是因数分解 $g$、basis
重建和所有步骤的 saturation certificate 都是必须实现的内容。$g$ 较小且已分解时可行，
当前没有项目原型或覆盖 shear/高阶的证明，不能先宣称它已彻底解决 saturation。

## G.4 候选算法比较

| 算法 | exact | saturated | int64-only 可证明 | Numba 可实现 | 系数增长 / 本项目判断 | 实施推荐 |
| --- | --- | --- | --- | --- | --- | --- |
| 当前 Euclidean + 显式 $V$ | 是，未溢出时 | 是 | 否；profile 中存储 A/V 已超 64 bit | 容易写但不安全 | quotient sequence 和 V 同步更新；换 `i128` 实现不满足目标 | 淘汰作生产核，保留 reference |
| signed / weighted union-find | 是 | 是，仅上述 $0,\pm1$ 两变量子类 | 是；每 component 值 $\pm1$ | 极易 | 84/165 profiled matrices 命中；一般多变量不适用 | 作为精确快速分支 |
| 有限群作用 / Reynolds $W$ | 是 | 否，image lattice 可能 finite-index | 有条件；要求 $gT$ 累加、$AW$ 与 projector certificates 有界 | 易到中 | 输出 generator 系数受群平均控制，避免 Euclidean $V$；需独立饱和 | 作为 exact rank/nullity 与 generator 分支 |
| fraction-free Bareiss | 是 | 还需 kernel transformation | 否 | 中 | stored pivots 有 minors bound，但 numerator 包含两乘积；若只用 int64 要求每个乘积/差都先证明，不是 exact division 自动安全 | 不推荐主核 |
| HNF / SNF 经典 Euclidean | 是 | 可 | 否 | 中 | unimodular transforms 与 minors 会 swell；canonical form不消除 word overflow | 不推荐主核 |
| Kannan–Bachem HNF/SNF | 是 | 可 | 一般不是 fixed int64；其界是输入 bit length 的多项式，不是 63-bit 上界 | 复杂 | [Kannan–Bachem 1979](https://doi.org/10.1137/0208040) 解决 bit complexity，不解决 machine-word 上界 | 不值得为本项目重写 |
| modular SNF / p-adic / Howell | 是，带重建证书 | 可，须正确 demodularize | residues 内是；lifting/transform 常引入 wide 或大 CRT | 高 | 适合限制 intermediate；模 $p^k$ 也要处理 composite zero divisors；不自动得到 int64 saturated basis | Howell 用于 bounded lattice saturation候选 |
| modular nullspace + CRT + rational reconstruction | 是，可 exact certify | 可，将 rational kernel 转 lattice congruence preimage | **在下述明确 safe domain 可证明** | 中高 | 两个约 $2^{31}$ prime 的 product 是 $4.6117\times10^{18}<2^{63}$；可用一次 64-bit CRT 与有界 RR，避免大 CRT accumulator | **最推荐通用 basis 方向** |
| Bareiss + 每行 gcd | 是，核保持 | 不保证饱和 | 否 | 中 | gcd 在中间乘法后才发生；除非先界住两个乘积仍不够 | 只作实验优化 |
| determinant/cofactor，nullity 1 | 是 | primitive vector 还需 gcd 归一 | 有条件 | 易到中 | kernel 向量由 rank minors 给出；Hadamard 界可证；FC5 onsite 的向量很小 | $d=1$ specialized fast path |
| float QR/SVD | 只对数值 subspace | 否 | int64 问题不存在 | NumPy/SciPy 更适合 | 不能当作 exact folded rank / exact lattice certificate | 仅 Cartesian projection与独立数值校验 |

HNF/SNF 的“多项式 bit complexity”允许整数随输入大小增长，完全不同于每个值必须小于
$2^{63}$；见 Kannan–Bachem 原论文摘要关于 polynomially bounded intermediate bit lengths。
现代 FLINT 同样用 multi-modular reconstruction 和 arbitrary-precision `fmpz` 实现 exact
HNF/SNF，不代表 Word-only 算法存在；其 API 说明 modular HNF 依赖行列式倍数，
一般 nullspace 也可能返回非最小整数 basis。[FLINT integer matrix 文档](https://flintlib.org/doc/fmpz_mat.html)
明确区分 residue arithmetic 与 `fmpz` 大整数 lifting。

Bareiss 第 $k$ 步存储的 fraction-free entries 是 minors 的比值；分子
$a_{kk}a_{ij}-a_{ik}a_{kj}$ 可被输入行范数 Hadamard bound 的两个 minor 乘积界定。
要求分子安全，须逐步证明这个乘积差自身小于 $2^{63}$，而不是只证明相除后的 minor 小。
对任意 matrix，$|minor_r|\le\prod_{i=1}^r\|row_i(A)\|_2$；constraint entries 上界
$\Delta=\max_g\|T_gE_p-E_p\|$ 时，粗界为
$|minor_r|\le (\sqrt m\Delta)^r$。剪掉重复行/使用 sparsity 可改善界，
但对 $m=81,r=66$ 的 shear 样本仍不能从输入元素范围自动推出 64 bit。
这路线无法消除 widening 需求。

每步 gcd normalization 保持同一 rational row space/kernel，但最危险乘积发生在归一化之前。
除非把每步乘积本身也改成可界 remainder arithmetic，否则不是溢出证明。
直接 cofactor kernel 是特定 nullity 的有效优化，cofactor 本身受 minor bound，
所以只在 bound 证明通过时运行；nullity 大时输出成本也失去优势。

## G.5 bounded chart 算法及整数范围证明

将精确 rank 与饱和 basis 分成两个数学步骤，不再同步追踪整个 unimodular $V$。

**第一步：在有限域选 rank 和 pivot chart。** 对 $A$ 约化模素数 $p<2^{31}$，
Numba row elimination 中每个 residue $0\le x<p$：

$$
xy\le(p-1)^2<2^{62},\qquad x-yz\in(-2^{62},2^{31})subset\mathbb Z_{64}.
$$

每次乘后立即 mod，故不用 Python/SymPy bigint。pivot rows/columns 只作为 chart 的索引。
单素数秩只是 rational rank 下界。若使用 G.3 projector identity，可先由精确 trace 得
$d=\dim\ker_{\mathbb Q}A$，然后任一模素数秩 $m-d$ 的 pivot chart 就有正确 rank；
否则使用独立 Hadamard distinct-prime rank certificate，且只累积 prime count/bits
作为界，不能在 int64 中直接乘无界的 primes product。
两个以上不同 primes 都要验证相同 rank/profile，bad primes 换掉；还须存在 exact upper
rank certificate，不能把“多素数观测相同”当证明。

**第二步：对固定 pivot block $B=A[:,P]$ 建 rational chart。** 对每一 free column $c$，
解 $Bx_c=-A[:,c]$。可用各模数上的 RREF，reconstruct 每个有理系数 $n/d$。
Wang rational reconstruction 在先有严格 $|n|\le N,d\le D$ 时，只需
$M>2ND$ 即能保证唯一；见 [Monagan 2004 的算法说明与引用的 Wang 定理](https://www.cecm.sfu.ca/~monaganm/papers/MQIRR.pdf)。
profile 取样的 chart 最大单项 $|n|d$ 为 $9.86\times10^{15}$，远小于本次 CRT
modulus 的一半 $2.3058\times10^{18}$。

取两个既定 31 bit primes
$p_1=2147483647,p_2=2147483629$，其乘积
$M=4611685975477714963<2^{63}-1$。Residues $a,b$ 的 CRT 用
$t=((b-a)\bmod p_2)(p_1^{-1}\bmod p_2)\bmod p_2$，然后
$x=a+p_1t$；所有因子小于 $2^{31}$，最后非负 $x<M<2^{63}$。
中心化时只做与 $M/2$ 比较及减 $M$，绝对值仍小于 $2^{62}$。
有理重建的 extended Euclid 只保留 remainder 与一个系数序列；每步 remainder
$q b$ 由 Euclidean invariant 保证不超过当前 remainder $\le M$，Bezout coefficient
绝对值不超过 $M/\gcd(M,u)$，所以临时及 stored values 都严格小于 $M$。
不追踪矩阵 $V$，也不把多 prime CRT 累积到两个 machine words 以上。

不预先知道 chart bound 时，采用受限 denominator windows
$D=1,2,4,\ldots,2^k$，$N=\lfloor(M-1)/(2D)\rfloor$，只接受满足窗口的候选；
逐个候选后做独立精确证书，window 扫描只到由入口确定的上限。
若不能唯一重建、pivot chart bad、整数界不通过或无法建立 rank certificate，
该输入不进入 unproven fast path，抛出有明确阶段信息的错误。不能让浮点近似挑选 basis。

**第三步：恢复饱和 integer kernel，而不计算大 CRT。** 逐列有理 chart 可写成
$K_{rat}=(-F/\delta;I_d)$，其中 $F\in\mathbb Z^{r\times d}$，
$\delta>0$ 为 common denominator。所有整数解对应

$$
L=\{z\in\mathbb Z^d:Fz\equiv0\pmod\delta\},\qquad
K_{sat}=(-F H/\delta;H),
$$

其中 $H$ 是上述 congruence preimage lattice 的整数 basis。证明是直接的：$z$ 是自由坐标，
$-Fz/\delta$ 为整数当且仅当 congruence 成立；故 $H$ 的任意整数 basis 映射为
$\ker_{\mathbb Q}A\cap\mathbb Z^m$ 的整数 basis，自动 saturated。
这个 lattice 含 $\delta\mathbb Z^d$，可将有限模组 $\mathbb Z/\delta\mathbb Z$
上的 kernel/module 用 Howell form 求出，再与 $\delta I_d$ 一起 lift 到 HNF。
Howell form 正适合有 zero divisors 的 composite $\mathbb Z/\delta\mathbb Z$，
不能把 composite 模数误当 field；[Fieker–Hofmann 的 modular normal form 工作](https://www.cambridge.org/core/product/identifier/S1461157014000291/type/journal_article)
说明在 quotient ring 内做 Howell/strong echelon 再 demodularize，可避免 characteristic-zero
消元。但本项目还需实现 Z modulus 下 congruence kernel 到饱和 lattice basis 的明确特例，
并证明 Howell 输出加 $\delta I$ 的 lift 生成完整 $L$；文献不是本仓库已验证代码。

**第四步：kernel transform 的 int64 dot products 使用 quotient/remainder accumulation。**
直接算 $FH$ 会有 $d$ 项乘积和。写每个 $F_{ij}=q_{ij}\delta+r_{ij}$，
$0\le r_{ij}<\delta$。每项 $F_{ij}H_{jk}/\delta$ 的 quotient/remainder 可用
$q_{ij}H_{jk}+\lfloor r_{ij}H_{jk}/\delta\rfloor$ 与
$r_{ij}H_{jk}\bmod\delta$ 求出。若 $|H_{jk}|\le\delta$，则
$r_{ij}|H_{jk}|<\delta^2$；取 $\delta<2^{31}$，这个乘积严格小于 $2^{62}$。
若每个最终 basis entry 的贡献数至多 $d$，sum accumulator 由
$d(\max|F|+2\delta+1)$ 界定。validator 对该式、$F,H,\delta,d$ 和输出 entry
分别检查 int64 max；每个中间 remainder 不累积到大整数。未证明的小 $\delta$、basis
entry 或 sum bound 时拒绝，不允许 wraparound。

这给出**一个有明确 admission test 的 int64-only exact kernel safe domain**：

1. rank/nullity 与 pivot chart 有 exact certificate；
2. 有理 reconstruction 的 $2ND<M$ 成立，且两 prime 独立 chart residues 相符；
3. $\delta<2^{31}$，Howell lift 的存储尺寸可用 `np.intp`；
4. $\delta^2<2^{62}$，$d(\max|F|+2\delta+1)<2^{63}$，
   其他实际乘积/offset/element count 的 preflight bounds 均通过；
5. kernel 满足 $AK=0$ 的 residues 证书及 rank certificate，因此 columns 数等于 exact nullity；
   saturation 来自 Howell congruence preimage construction，而非仅凭 annihilation。

这个是数据依赖的域，不是只看 $N,P,K,S$ 的闭式通用保证。入口可用有界算术计算
bit-length upper bounds；取得压缩 constraint/action 后只做一次局部 exact-domain validation，
Numba elimination 内不用逐操作 checked arithmetic。
若业务要求对每一个原 Python bigint 可表示输入都接受，单一固定 pair primes 不可能
reconstruct 任意大 rational numerator/denominator；那是输入域改变，不是代码技巧可以消除。

profiling 的参考 rational charts 在 27,952 个 entries 上做了两 prime modular RREF，
再用精确有理 chart 作为 oracle 逐项比较 reconstruction：全部相同。$M$ 如上，
测试中最大有理分子 × 分母 $9.86\times10^{15}$，最大 common denominator
$680,581,440$，最大 numerator-normalized $F$ 为
$2.2487\times10^{15}$。按 quotient/remainder bound，profile 最大
$d(F+2\delta)$ 是 $3.3731\times10^{16}<2^{63}$。
Rational reconstruction script `/tmp/mlfcs-chart-check.py` SHA-256
`b0639723e003b1708ba2a34d61261d5155b1761207bb853619c4f8f75432f51e`；结果
`/tmp/mlfcs-chart-check.json` 报告 matrices=165、entries=27952、exact_matches=true、
max_RR_intermediate=$M-1$。注意这是使用真实 rational chart oracle bounds 的算法
可行性差分，不是从零完整实现的 production kernel，也没有实现 Howell lift。

## G.6 八类方案逐项结论

### A. Euclidean elimination 与 scale-separated 表示

`q*b` 超界可以用 remainder-first 乘减避免无谓乘法 overflow：先求 $bq$ 对某 modulus
的 residue，或作 32-bit limb 拆分。但前者仅给 residue、后者实质 software-wide integer；
两者都没有阻止 A/V stored coefficients 超界。本项不选。

### B. Fraction-free / Bareiss

精确除法降低 stored minors 的增长，却不保证 numerator 两个乘积或乘积差落在 int64；
按当前稀疏行上界可得到逐 pivot 充分条件，Shear31/R=1023 的四阶 tensor 已使
粗界远超 word。可以以后按真实 minors 选择性跑，不做通用核。

### C. HNF

变换版本维护的 unimodular multiplier 本身可增长；不变换 HNF 也要做精确行列组合。
把当前 Euclidean loops 贴上 HNF 名称并不改变 widening。经典和 polynomial bit
complexity 算法都允许任意 word 外系数，不满足硬 int64 目标。

### D. SNF

SymPy SNF 是 Python reference/oracle，也是现有 saturation 语义来源；不是生产迁移方案。
模素数分解、p-adic valuation、Iliopoulos / Kannan–Bachem modular SNF 能在 residue
域中处理 large input，但恢复 elementary divisors、multipliers 或完整 lattice basis
时通常需要 CRT/multiword 或另一个已证明 bound。若只要 rational span/rank，SNF 过度计算。

### E. Modular nullspace / rational reconstruction

是最有希望的通用路线，因为本项目后续真正消费 rational span，matrix 的 rank 已经有
modular certificate 技术；saturation 可经 rational pivot chart 转为 composite modulus
congruence preimage，而不是重建 arbitrary unimodular $V$。profiling 的实际图表满足两
31 bit primes 的有界重建，shear31 也通过；最强风险是真实合法 chart denominator 或
numerator 超 admission bounds、坏 prime、Howell lift实现复杂。应先做离线独立原型，并以
SymPy saturation 和精确 kernel 检查为 oracle，再决定 safe domain 能否覆盖全部材料案例。

### F. Rational chart / common denominator

在给定 nonsingular pivot minor $B$ 下 Cramer 法给
$\delta=|\det B|$、$F=-\operatorname{adj}(B)C$，故每个分子、分母受 $r$ 阶 minors
Hadamard bound约束。但直接 determinant 的乘法中间值可能越界；应从 modular determinants
恢复或直接 modular rational reconstruction。选择 low-norm/sparse pivots 显著有利；
chart search 不能无证书地换到近似最优 pivot。

### G. 每步 gcd 与 primitive row

row gcd 保持 rational row span，不能保证 kernel lattice saturated（还需 basis saturation）。
并且 gcd 在乘法后发生，不保护乘积 intermediate。对低系数实际样本有助减小存储量，
Shear31 起始 gcd 归一也不能消除大坐标表示；不是总算法证明。

### H. Cofactor / low nullity

nullity 1 时由 rank minor 构造一个 kernel vector，除以全分量 gcd 得 primitive basis；
向量有 Hadamard upper bound，可先判 int64 safety。Si onsite FC5 的 exact chart 是
$(-2/15,1)$，saturated integer generator仅最大 15，而 current Euclidean V 长 170 bit；
这是无需追踪 V 的强 specialization。nullity >1 时需要兼容 lattice saturation 的多列
construction，不宜将 cofactor 扩充成所有情况。

## G.7 推荐实现路线与可证明边界

推荐分层实现，但仍统一 Numba exact backend：

1. 压缩相同 stabilizer/tensor actions，建立 int64 constraint blocks；入口检查
   $S,p!,3^p,R,H$ 与 row/action coefficient bounds。
2. 对 rows 经归一后 `nnz <= 2 && abs(c0)==abs(c1)` 的块用 signed union-find，
   输出 component primitive basis，精确覆盖该子类。
3. 对完整有限群/已验证 projector block 优先用 $W=\sum T_g$ 求 rank/nullity 和 bounded
   invariant generators；modular residues 证明 $AW=0,W^2=gW$，trace/g 给 exact rank。
   $d=1$ 且 primitive vector可由 gcd 归一时直接返回饱和 basis。
4. 通用剩余块用 Numba modular pivot + two-prime rational chart；small-domain composite
   Howell kernel 求 $H$ 并 quotient/remainder lift saturated basis。所有 Python `Fraction`/
   SymPy 只在研究/测试 reference，不进入生产。
5. Folded rank 只求 rank，不构造 integer kernel。把 sparse folded coefficients 分 prime
   reduction 累加；以实际系数和 row-norm Hadamard bit bound 决定所需 prime 数；每个 prime
   保持 word-sized residues。对 folded rank 可避开 float 与大 basis CRT。
6. 对无法给出 pivot/reconstruction/modulus/lift bound 的输入抛出 OverflowError，错误需写清
   不满足的是 chart 或 saturation word bound。不要静默切回 Python bigint；按用户最后的
   硬目标，unsupported case 要成为显式 safe-domain 边界。

理论 unrestricted domain：**不存在**覆盖任意整数矩阵/任意输出系数的固定 int64 精确算法。
例如输入 $A=[1,-h]$ 的 saturated kernel generator 是 $(h,1)$；令 $h>2^{63}-1$，
输入/输出本身已超 int64。更关键地，即使输入元素极小，任意稀疏 integer relation 链
也可有指数型 primitive kernel coefficient。因此 finite-word safe domain 必须约束输入与
输出/算法 certificates；换算法不能打破信息论上的输出范围。

本项目合法 safe domain：已能写出相当实用的**动态充分域**（G.5），它能包含当前 profiled
shear31/FC4 与 onsite FC5 chart oracle，而不是只限 $R=1$。尚不能声称由现有
$B_{total},S,R,H$ 全局上界推出每个合法 cluster 都一定通过：最坏 minor/Hadamard
bound太松，且一般 rational chart output 取决于实际 pivot minors。入口认证可以严格决定
每个实际 kernel 是否落入动态域；域外明确拒绝。因此它满足“在被证明的项目 safe subdomain
中 Numba int64 exact”，还没有证明“所有原 Python/SymPy 接受的 mlfcs 输入都能接受”。
要升级成后者，需证明 spglib-generated finite-group lattice representation 对每个允许
$p$ 的 saturated basis 有 tight uniform coefficient bound；本次 profiling 和 Reynolds
identity本身不足以推出该结论。

现实 workload：所抽样 165 个块都远在 int64 动态界内（chart denominator最大
680,581,440；kernel transform保守 accumulator上界最大 $3.38\times10^{16}$），
且两素数重建全部成功。剪切31 FC4 是最接近 general-case stress 的实例，仍通过；
Si FC5 则展示旧 algorithm 巨幅假性 growth，但新 rational chart 与最终 exact generator
很小。数据支持继续原型开发，不代表 benchmark 已完成，也不代表集成代码可发布。

综合方案表：

| 算法 | exact | saturated | int64-only 可证明 | Numba 可实现 | coefficient growth | 实现复杂度 | 推荐 |
|---|---|---|---|---|---|---|---|
| 当前 Euclidean + $V$ | 是 | 是 | 否 | 是 | 已实测 A131/V170 bit | 低 | 仅 oracle |
| signed union-find | 是 | 是（命中子类） | 是 | 极易 | 无，输出 $\pm1$ | 低 | fast path |
| Reynolds + direct orbits | 是 | 一般否 | 有 bound 时是 | 是 | $gT$ | 中 | rank/d=1 fast path |
| Bareiss | 是 | transform另需 | 否 | 是 | minors；乘积先涨 | 中 | 否 |
| HNF/SNF Euclidean | 是 | 是 | 否 | 是 | multiplier可涨 | 中 | 否 |
| modular SNF | 是 | 可 | residues是，lifting未普遍 | 是 | CRT/prime powers | 高 | 不优先 |
| modular chart + CRT/RR + Howell | 是 | 是（preimage证明） | dynamic domain是 | 是 | primes与$\delta$有明确上界 | 高 | **主路线** |
| QR/SVD | 浮点 subspace | 否 | 不相关 | 外部 LAPACK | 无整数增长 | 低 | 仅 Cartesian |

## G.8 实施前要完成的独立研究关口

本阶段没有改 production code，以下是算法进入核心前的必要技术关口，而不是扩大分析范围：

- 独立实现 modular chart + RR + composite Howell preimage，至少覆盖 165 个记录实例；
  每个 saturated basis 对比原 SymPy `saturated_kernel` 的 rational span、index 1、$AK=0$。
- 对自定义 symmetry 验证 projector/action identities；对 spglib 输入验证 action image 去重与
  轴置换 quotient 的等价性，尤其 lattice shear、大平移 site shifts 与 repeated labels。
- 证明 two-prime reconstruction 选择的每个 chart 唯一；bad primes 与 denominator primes
  必须能动态剔除；固定 prime list用尽必须 fail closed。
- 对 Howell lift给完整不变量、rank和模数合成证明；证明有限模 module kernel 加
  $\delta I$ 恰为 integer congruence preimage，而不是仅同一 rational span。
- 完整材料案例覆盖 FC2–FC4；对 FC5/FC6 先只跑可承受 cutoff 的 representative，
  单独记录所有 intermediate bit bound、shape/bytes 与最终 basis，并与 stash Rust 和原 Python
  差分。Rust stash 始终作为 reference，不 pop。
- folded rank 设计一个 residue-only exact certificate；Aliasing/rank 不再走
  integer_kernel。Cartesian basis 的最终浮点 QR 与 exact kernel rank 分开验收。

数学研究结论足以选定候选方向，不足以宣告 kernel 迁移已经完成。现阶段主推荐是
**signed union-find 子类 + modular rank/projector + modular rational chart + Howell saturation**；
其中最需原型证实的是 general Howell lift 与从 $A$ 的实际项目 safe domain导出的
admission coverage。标准 `int64` 可在明确定义且可预检的域内做到完整 exact saturated
kernel；无界 arbitrary-precision 输入与无界 coefficient 输出不能由固定 int64 全覆盖。
