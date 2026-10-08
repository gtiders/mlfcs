# 初始化阶段的 ASR 自由坐标

## 实施范围

本次实现位于独立 worktree `/home/gwins/Documents/mlfcs-asr-nullspace`，分支
`asr-nullspace-validation`。原工作目录的未提交实现已先做独立快照；没有覆盖原目录。

```python
space = ClusterSpace(primitive_atoms, cutoffs={2: 3.0, 3: 3.0, 4: 3.0}, asr=True)
```

`asr=False` 为默认值。`space.n_parameters` 始终是原始物理分量参数数目；
`space.n_free_parameters` 是拟合所用自由坐标的数目。

原有每个 orbit 的整数核、Cartesian basis、观察分量及其参数布局保持原语义。
全局 ASR 不调用局部整数核来生成巨大的稠密基，也不使用 SymPy、bigint 或宽整数生产消元。

## 坐标定义与精确证书

第 $p$ 阶 orbit 的 lattice basis 为 $B_o$，张量坐标变换为

$$
T_p=(A^{\mathsf T})^{\otimes p}.
$$

令 $E_o$ 选择现有 Cartesian 参数定义的观察分量，则

$$
W_o=E_oT_pB_o,\qquad \theta_o=W_oc_o.
$$

每个 $W_o$ 都是小的可逆 orbit-local 矩阵。全局不物化 block-diagonal $W$。
通过 lattice rotations 和 tensor-axis permutations 展开 $B_o$，固定前 $p-1$ 个位置并
对最后位置求和，得到稀疏整数 ASR 矩阵 $C$。
Cartesian ASR 与 $Cc=0$ 等价，因为张量坐标变换对同阶所有求和项相同且可逆。

两个小于 $2^{31}$ 的素数使用相同 pivot schedule 作稀疏模消元。
模乘积小于 $2^{62}$；逐操作取模防止部分和增长。
重建每行有界有理数因子并证明

$$
C=LU.
$$

证书使用行公共分母、先除后乘的 LCM 和实际乘积/累加的 int64 检查。
两素数 rank 一致本身不是证书。$U$ 的 pivot 为单位有理数且前序 pivot 列为零，
所以其行独立。对每个 pivot 序号，$L$ 都有一行最大非零列恰为该序号，
这些行形成非奇异三角子矩阵。因此 $L$ 满列秩，精确因子等式证明

$$
\operatorname{rank}_{\mathbb Q}C=\operatorname{rows}(U),\qquad
\ker_{\mathbb R}C=\ker_{\mathbb R}U.
$$

任何未能重建、认证或满足实际机器字范围的实例都报错；不以浮点 rank 或旧 bigint 路径兜底。
分母准入沿用已有有理重建机器字域，不能覆盖任意整数矩阵。

把列分成 pivot 与 free 后，保留稀疏三角方程

$$
U_Pc_P+U_Fc_F=0.
$$

自由变量是实数 $c_F$，通过倒序代入求 $c_P$，再由 $W_o$ 转为原始物理系数。
这里所需的是完整实数子空间，而不是全局 saturated integer lattice basis；
并未降低原有 orbit 整数核的语义。

证书阶段结束后丢弃 $L$、两个模因子和原始约束矩阵；长期只保留
CSR $U$、pivot/free 列和小矩阵 $W_o$。证书不是运行时随每个调用传递的门票。
热路径中的三角求解及物理张量运算使用 float64；精确离散代数仍使用 int64。

### 与原实施设想的差别

当前排序为非零数较少的行优先，新 pivot 优先原始 incidence 较低的列。
这是明确的稀疏度启发式，并非完整的动态 Markowitz 排序。
不同连通分量的列天然不会相互消元，当前不额外物化二部图来排序。
Ba 验证通过，但更大模型的填充仍可能严重；动态排序是下一步性能优化，不能宣称已完成。

## 拟合接口

```python
mapping = ClusterMap(space, supercell_atoms)
dataset = ForceDataset(mapping, structures)
raw = FitSystem(dataset, representation="raw")
model = raw.solve()

# 外部求解器读取的就是未经列归一化的自由坐标方程。
eta = scipy.linalg.lstsq(raw.design_matrix, raw.forces)[0]
model = raw.force_constants(eta)

normal = FitSystem(dataset, representation="normal")
model = normal.solve(rtol=1e-10, maxiter=1000)
```

当 ASR 启用时，公开 raw 矩阵为 $A_{\rm free}=A_{\rm physical}WT$，
其中 $T$ 是隐式三角 lift；normal 方程为其 Gram 矩阵与右端项。
归一化仍只属于 solver 临时操作，公开数组不归一化。
求解器仍只有原来的 raw dense least squares 和 normal MINRES。
返回的 `ForceConstants` 保存 canonical physical coefficients，而不是自由坐标。

原始力设计按最多 64 行分块构造，再通过三角 lift 的伴随转换成自由坐标行。
不分配原始整帧设计和全局 $n\times d$ 基。
`ForceDesign.matrix()` 继续提供原始物理设计，供原有模型评价及外部直接使用。
`FitSystem` 才暴露受约束方程。

`FitSystem.residual(model)` 要求模型位于准备好的 ASR 子空间；不允许只提取 free
coordinates、悄悄忽略一个不满足约束的模型的其余分量。

## 有限差分接口与外推

```python
sampling = FiniteDifference(mapping, order=4, disps=(0.03, 0.05))
frames = sampling.displacements()
# 外部计算器按 frames 顺序计算并保存 forces。
dataset = ForceDataset(mapping, calculated_frames)
model = sampling.reconstruct(dataset)
```

采样数量与顺序不改变，也没有重新引入 frame metadata。
先按原有 central stencil 求导并作 even-error extrapolation，随后把全部原始观察结果
通过隐式 lift/adjoint 的受约束最小二乘映射到 ASR 子空间。
因此当前获得的是约束重建优势，**还不是更少的差分样本**。
精确多项式测试覆盖 FC2/3/4 双步长外推；有噪声时结果与单独逐 orbit 重建可能不同。

有限差分的受约束最小二乘使用矩阵自由 LSMR；它不是新增的 `FitSystem` 求解算法。
不构造自由坐标到全部观察量的稠密矩阵。
当前仍要求超胞保留原始参数的完整 folded rank；仅保留 ASR 子空间的更小超胞支持尚未实现。

## 内存约束与适用范围

不设置人为内存预算。实际分配失败、数组容量或算术范围失败照常报错。
CSR 与隐式三角表示避免全局 $n\times d$、$d\times d$ kernel workspace，
但稀疏消元仍可能填充，不能用输入 CSR 大小推断初始化峰值。

raw 数据仍占 $O(N_{\rm eq}d)$，normal 数据仍占 $O(d^2)$。
Ba 本次自由维数为 $166805$，仅一个 float64 normal 矩阵仍约 207.3 GiB；
ASR 初始化不能解决这种 normal 表示的内存问题。
后续若支持大 Ba 拟合，需要另行设计 matrix-free fitting；本次不新增第三条拟合路线。

## 生命周期与 IO

准备结果由 `ClusterSpace` 持有，质量修改可复用，因为 ASR 不依赖原子质量。
原来的 HDF5 继续只保存原始参数化模型；不把 solver layout 或稀疏因子变成持久格式。
`ForceConstants.load()` 恢复 canonical 模型，恢复的空间没有预备 ASR 因子。
声子、Ewald 局部修正、旋转约束、现有后施加 ASR 均继续消费原始参数布局。

## 验证与后续

1. 已验证原有 fitting、finite-difference、invariance 与整数核基线测试。
2. 对 Si 和 unimodularly sheared Si 检查 exact factors、rank/nullity 和 Cartesian ASR。
3. 对 Ba 的 FC2/3/4 检查稀疏因子、实际内存与 Cartesian ASR，未运行热导率。
4. 对 raw/normal 拟合、lift/adjoint、分块设计、FC2/3/4 外推、质量拷贝和 native IO 建立测试。
5. 下一步优化证书临时字典内存与 pivot 排序；其收益应以串行测量证明。
6. 差分减少采样、更小受约束超胞和矩阵自由大规模拟合各自独立推进，不能混称为本次已完成。

## 串行测量记录

本节保留首次实现 `00df226` 的历史记录；当前改进后的测量见后续章节。

Ba 使用 54 原子 primitive、cutoffs 为 6/6/3 angstrom、body orders 为 2/3/4。
测试环境为 Python 3.12、Numba 0.68、NumPy 2.5；正式初始化测量设单线程，
Numba 磁盘缓存已经预热。以下不是 cold-JIT 指标。

| 模型/阶数 | 原始参数 | ASR rank | 自由维数 | 保留的因子与 orbit maps | Cartesian 相对残差 |
|---|---:|---:|---:|---:|---:|
| Si FC2 | 3 | 1 | 2 | 小型 | $4.36\times10^{-17}$ |
| Si FC3 | 5 | 2 | 3 | 小型 | $4.06\times10^{-17}$ |
| Si FC4 | 14 | 10 | 4 | 小型 | $4.57\times10^{-16}$ |
| Ba FC2 | 9567 | 483 | 9084 | 1.14 MiB | $1.44\times10^{-17}$ |
| Ba FC3 | 184554 | 28213 | 156341 | 62.03 MiB | $1.04\times10^{-16}$ |
| Ba FC4 | 9642 | 8262 | 1380 | 2.68 MiB | $1.01\times10^{-16}$ |

剪切等价 Si 的三个 rank/nullity 相同，Cartesian 相对残差最大约
$1.16\times10^{-15}$。Ba 重建后的三角因子最大系数分别为 1、4、1。

Ba 普通初始化约 6.32 秒；直接 `ClusterSpace(..., asr=True)` 初始化为 31.99 秒，
该阶段累计峰值 RSS 868868 KiB（约 0.83 GiB）。三个准备结果长期数组合计约 65.86 MiB。
这里的 RSS 包含 Python、Numba、原始 cluster model 和初始化临时对象。

完整 differential 脚本另行物化现有 Cartesian ASR 矩阵以验证残差，
因此脚本总峰值可达约 3.65 GiB。**该数字不能当作生产初始化峰值**。
此前独立因子实验的总 RSS 同样包含这一验证工作区。

记录文件为 `research/asr_nullspace/results.json`。复现命令：

```bash
NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python research/asr_nullspace/validate.py ba --initialize-asr
```

该命令包含独立 Cartesian 检验；生产初始化指标对应其 `stage=initialization` 行。

最终测试命令：

```bash
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 uv run pytest -o addopts='' -q
```

结果：420 passed，21.60 秒。告警来自现有 ASE/spglib 兼容性接口等。
新增测试同时覆盖独立有理数 oracle、随机稀疏矩阵、空方程空间及溢出拒绝。
Ruff 检查、格式检查和 `git diff --check` 均通过。

基线另有一项日志测试仍期待 raw 使用 LSMR；当前基线源码实际已使用 direct least squares。
本次将该测试期待改为 `algorithm=least_squares` 与 rank 日志，未修改求解算法。

## Matrix-free 验证与初始化工作区改进

### 验证复用现有算子

当前验证脚本使用已有 `AcousticSumRuleOperator`，通过 `matvec` 求 Cartesian ASR
残差，通过 `max_row_abs_sum` 计算与原显式矩阵相同的残差尺度。
不再物化大型 Cartesian ASR 矩阵；小规模测试仍独立对照显式矩阵。

测试脚本测量的是完整进程累计峰值 RSS，初始化指标在构造 `ClusterSpace` 返回时
立即记录，后续验证指标另记。验证为统计整数 CSR 大小仍重新组装了小型整数约束，
但这些数组在构造验证算子前释放；该统计工作不属于生产初始化。

### 初始化怎样采用流式方法

当前精确三角因子算法需要访问整数方程行与 pivot 系数，不能直接使用
float64 `AcousticSumRuleOperator` 的乘法接口替代。这里没有声称任意精确算法
都必须物化矩阵；采用 black-box exact algebra 会是另一项算法设计。

现有方案中，Ba FC3 的输入整数 CSR 仅约 12.23 MiB。主要优化如下：

1. 每个素数消元结束后立即把 $U$ 和模 $L$ 压成 CSR，释放该素数的哈希表，
   再开始下一个素数。两个完整消元字典不再同时保留。
2. 有理重建在 Numba 中逐行处理压缩数组，避免把每个因子元素装成 Python 标量
   或新的有理因子字典。
3. 有理数 $L$ 不整体物化；其每一行重建后立即用于精确 $C=LU$ 认证并释放。
   两个模 $L$ 仍保留为压缩残差数据，认证所需的 $U$ 则仍显式保留。
4. 最终因子直接复用重建后的压缩数组，不经过大型 Python 列表再次打包。
5. 约束组装使用 NumPy 数组分块，减少逐元素 Python 对象。
6. 相同 lattice basis 和 observation rows 定义相同的 orbit-local 物理坐标变换。
   在单次同阶组装内共享这些只读矩阵，保持每个 orbit 的系数块独立；
   不引入全局缓存或共享参数。

数学算法、两个素数、动态重建范围及逐乘积/累加证书保持原语义。
没有增加浮点 rank fallback、bigint 路径或人为内存预算。

### 同条件串行测量

单线程、Numba 磁盘缓存已预热，模型仍是 Ba FC2/3/4，cutoffs 为 6/6/3 angstrom，
body orders 为 2/3/4。未运行热导率。

| 实现 | 初始化时间 | 初始化峰值 RSS | 完整 matrix-free 验证峰值 RSS |
|---|---:|---:|---:|
| 改进前，验证已换成算子 | 31.34 s | 861064 KiB，0.82 GiB | 920084 KiB，0.88 GiB |
| 改进后，第 1 次 | 12.64 s | 541036 KiB，0.52 GiB | 576340 KiB，0.55 GiB |
| 改进后，第 2 次 | 12.95 s | 540816 KiB，0.52 GiB | 576504 KiB，0.55 GiB |

初始化峰值减少约 37%，耗时减少约 59%。这些是当前机器的 warmed-JIT 测量，
不属于 cold compilation 性能，也不保证其他晶体有同样的稀疏填充与共享比例。

| 阶数 | orbit 物理变换数量 | 独立存储的变换数量 | 当前长期数组 |
|---|---:|---:|---:|
| FC2 | 1081 | 2 | 0.50 MiB |
| FC3 | 7554 | 4 | 26.64 MiB |
| FC4 | 330 | 4 | 0.44 MiB |

长期数组按不同 ndarray 对象计数，不把多个 orbit 对同一矩阵的引用重复相加。
这些数字不包含 Python 对象、Numba runtime 或原始 cluster model；RSS 已包含进程开销。
三阶 rank 仍为 28213，自由维数仍为 156341，Cartesian ASR 相对残差约
$1.04\times10^{-16}$。Si 与剪切等价 Si 的 rank/nullity 及物理残差也保持一致。

完整回归结果为 **423 passed**。新增检查覆盖两素数中不同的非零位置、
被破坏的因子不能通过证书，以及相同物理变换的只读共享。

### 剩余内存来源

初始化仍需要原始 cluster model、当前素数的稀疏消元哈希表、压缩模因子、
原始整数 CSR 与已完成阶数的准备结果。稀疏填充仍可能增长；矩阵共享收益取决于
实际重复程度。把 12 MiB 的 CSR 换成 float64 乘法算子，不能消除精确因子工作区。

后续若继续压缩，需要针对消元哈希表或稀疏 pivot 顺序优化，并重新测量。
本轮没有改动 raw/normal 拟合算法，normal 的自由维数平方内存限制仍然存在。


## 自顶向下的 ASR 模块职责审查

本节在现有证明和内存测量基础上整理职责，没有更换消元算法、准入范围或物理参数布局。

### 调用链和归属

```text
ClusterSpace(asr=True)
  -> cluster_space.asr.prepare_acoustic_coordinates
  -> cluster_space.acoustic: lattice equations and physical coordinate maps
  -> cluster_space.acoustic_factorization: modular factors and exact certificate
  -> AcousticCoordinates
       -> FitSystem: restrict physical force rows and lift solved coefficients
       -> finite_difference: extrapolate observations, then constrained reconstruction

ForceConstants.enforce_rotation()
  -> force_constants.rotation
  -> cluster_space.asr.acoustic_constraint_matrix (FC2 only)
```

| 模块 | 职责 | 主要计算方式 |
|---|---|---|
| `cluster_space/asr.py` | Cartesian ASR 方程与矩阵无关算子，初始化准备入口 | Python 组装，Numba 正向、转置和残差尺度循环 |
| `cluster_space/acoustic.py` | lattice ASR 方程、orbit 坐标映射、自由坐标对象 | NumPy/SciPy 组装，Numba 合并重复项 |
| `cluster_space/acoustic_factorization.py` | 稀疏模分解、有理重建、精确证书、三角坐标变换 | Numba 数值循环，Python 分阶段调度 |

ASR 的方程、参数空间算法和初始化准备完整归 `cluster_space`。只保留初始化 null space
这一条 ASR 生产路线；`force_constants/asr.py`、`ForceConstants.enforce_asr()` 和投影
报告类型已删除，不保留兼容别名。`ClusterSpace` 不依赖上层 `ForceConstants`。
构造函数使用普通模块导入调用准备入口，无需局部导入。

### 表示与算法没有重复

Cartesian 算子作用于物理参数，用于独立残差验收和旋转约束；lattice 方程作用于整数 orbit
坐标，用于精确证明 rank 和实数零空间。两者实现相同 ASR 的不同坐标表示，不能把 float64
矩阵无关算子直接当成精确整数消元输入。显式 Cartesian CSR 只服务于 FC2 旋转投影和小矩阵
测试；高阶残差验收与 Ba 验证使用矩阵无关算子。

自由坐标只保证完整实数零空间，orbit 的 saturated integer kernel 继续由
`cluster_space/integer_kernel.py` 提供。有限差分的采样与步长外推不改变，ASR 在外推之后
使用同一自由坐标接口进行重建。

### Numba 和内存边界

Numba 加速模消元、因子压缩、两 prime 重建、逐行精确证书、三角 lift/adjoint，以及
Cartesian 矩阵无关算子的正向和转置循环。对象生命周期、组装、NumPy 线性代数与 SciPy
迭代调度保留在 Python 层。模块拆分保留 `njit(cache=True)`，没有引入 Python 消元替代路径。

稀疏填充和每 prime 临时 hash 表仍是初始化的主要内存来源。稀疏全局方程没有消失；不能
把当前实现称为完全矩阵无关的精确初始化。此前约 0.52 GiB 的生产初始化与约 0.55 GiB
的矩阵无关验证峰值仍按不同阶段记录，重组模块不改变其理论内存规模。

### 接口影响

`ClusterSpace(..., asr=True)`、`acoustic_coordinates(order)`、`n_free_parameters`、
拟合与差分入口保持不变，ASR 后处理入口删除。内部数值接口使用明确名称
`lattice_acoustic_equations`、`prepare_coordinates` 和 `factor_acoustic_equations`；
算子与显式方程的源码导入位置改为 `mlfcs.cluster_space.asr`，不保留旧位置的兼容别名。


### Null-space 单路线验收

删除 ASR 后投影后，完整回归为 **421 passed**。原后投影专属检查删除，新增初始化
多阶物理残差检查；旋转检查改用准备好的自由坐标构造模型，保留原有物理误差阈值。
小矩阵的 dense lift 仅作为测试 oracle，不进入生产初始化。

Ba FC2/FC3/FC4、cutoff 6/6/3 Å、body 2/3/4 的串行 warm-JIT 复核：初始化
**12.596 秒**，初始化与整段无矩阵验证的峰值均为 **539816 KiB，约 0.515 GiB**。
每阶 rank、自由维数、保留因子 nnz 和独立 Cartesian 相对残差与上一版本一致。
最高相对残差为约 $1.04\times10^{-16}$。本次没有拟合 Ba 或计算热导率。

教程代码已迁移到初始化 `asr=True`，移除投影报告和后处理调用。六份 notebook 的历史
执行输出逐项保留，开头明确标注尚未按新接口重新执行，不能将其误作本轮数值验收。


## 旋转修正复用初始化 ASR 子空间

本节更新前一阶段的模块布局与接口；以上性能、残差和源码布局记录保留为历史结果。

### 统一表示与物理参数度量

令认证三角因子为 $U$，orbit-local 物理坐标映射为 $W$，则

$$
\theta=Wc,\qquad Uc=0.
$$

因此 ASR 零空间在物理参数空间中的法向行空间由 $U W^{-1}$ 张成。
`AcousticCoordinates.normal_basis()` 对其转置执行 reduced QR，返回正交法向列基 $Q$。
法向数量来自精确因子证书；QR 仅作浮点正交化，不做第二次声学 rank 判定。

旋转矩阵 $C$ 使用相同长度归一化和几何截断，实际求解矩阵为

$$
E=C(I-QQ^T).
$$

若 $E$ 的保留 SVD 为 $L\Sigma V^T$，则修正为

$$
\Delta\theta=V\Sigma^{-1}L^T C\theta.
$$

修正位于初始化 ASR 子空间，并保留最小物理参数改变量的欧氏度量。不能直接把
$C K$ 的普通最小二乘解用于非正交的自由坐标 $K$，否则最小化的会是自由坐标长度。
旋转涉及实际原子间距，仍使用浮点线性代数；ASR 模分解和整数证书保持原实现。

### 删除与职责

删除 `cluster_space/asr.py`，包括独立 Cartesian ASR CSR 构造函数与
`AcousticSumRuleOperator`。删除其交叉验证测试、研究脚本中的 Cartesian 残差复核，
以及 `RotationResult.acoustic_before/acoustic_after` 和相应日志。
保留初始化必需的 lattice ASR 方程、精确 rank/nullspace 证书和数值范围检查。

当前调用链为：

```text
ClusterSpace(asr=True)
  -> cluster_space.acoustic.prepare_acoustic_coordinates
  -> lattice acoustic equations
  -> acoustic_factorization: certified sparse triangular factors
  -> AcousticCoordinates
       -> fitting / finite_difference: lift and adjoint
       -> rotation: normal_basis -> restricted rotational SVD
```

旋转只接受已准备 ASR 坐标的 ClusterSpace，未设置 `asr=True` 时明确报错；不隐式启动
另一条准备路线。原生文件只存 canonical 模型，加载不会自动恢复准备坐标；若需要加载后
进行旋转修正，需用相同几何、cutoff 与 body 设置重新构造 `ClusterSpace(..., asr=True)`，
再将加载模型的系数绑定到该空间。已有模型的 ASR 残差不再检测或修补。

### 内存边界

没有构造完整自由坐标 basis。法向表示占用 $n r$ 个 float64 元素，其中 $n$ 是 FC2
物理参数数，$r$ 是已认证的 ASR rank；QR 另有同阶工作区。Ba 对应
$n=9567$、$r=483$，单个法向数组约 35.3 MiB。此数组仅在旋转修正时临时构造，
不增加 ClusterSpace 初始化和保留因子的常驻存储。该路线仍需要稠密法向 QR，不能
称为完全无矩阵的旋转算法。


### 本轮验收

删除 ASR 交叉验证后，剩余完整测试为 **416 passed**。旋转物理矩、幂等性、
高阶系数保持，以及拟合和有限差分工作流均通过。没有新增独立 ASR oracle 或残差
复核路径。此前 Cartesian 相对残差与矩阵无关验证内存仅作为历史测量保留。
