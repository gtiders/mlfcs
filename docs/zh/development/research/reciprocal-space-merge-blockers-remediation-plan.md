---
title: 倒空间约化第二阶段物理修复与合并计划
audience:
  - advanced
  - developer
status: research
code_verified: 4.0.0a6
localized_only: true
---

# 倒空间约化第二阶段物理修复与合并计划

## 0. 文档定位、基线与实施顺序

本文替代旧的“倒空间约化合并阻断修复计划”。旧计划针对 `16ce266` 之前的
little-group 假阳性、位置 gauge、全星展开和性能热点；这些主体工作已经在
`reflector` 的 `1ee4f43` 上完成。本文只规定审查后仍需完成的第二阶段工作，不能把
已经完成的 P0–P7 重新实现一遍。

审查基线为：

```text
dev        3b3fa1a
reflector  1ee4f43
merge-base 3b3fa1a
```

实施顺序固定为：

1. 先在独立分支完成 `structure-relation-symprec-refactor-plan.md`，合回 `dev`；
2. 将 `reflector` rebase 到新的 `dev`，解决 `integer_lattice.py`、SSCHA 调用和公共导入冲突；
3. 在 rebase 后的新基线上实施本文；
4. 完成物理、门禁、教程和文档验收后，才把倒空间分支合入 `dev`。

不得在带有用户未提交修改的主工作树直接实现。研究文档目前可能未跟踪，实施者必须按
绝对路径读取，不能因为新 worktree 中看不到它就忽略规范。

## 1. 当前结论

### 1.1 已完成且必须保留

以下实现方向已经通过代码和聚焦测试审查，不得回退：

- `mlfcs.reciprocal` 是声子倒空间、SCPH 和 SSCHA 的单向叶子包；主线包不依赖它；
- 频率路径只对星代表点做本征求解，再按星展开频率；
- SCPH 在代表点构造协方差，逐星成员施加空间群作用、反幺正共轭和 stored-label gauge；
- `expand_star_matrices` 使用 reduced label 对应的位置 gauge；
- 全星门禁比较每个成员的直接动力学矩阵和代表点展开结果；little group 只保留为局部诊断；
- SCPH 的初始 FC2、每轮更新 FC2 和最终 FC2 都经过全星验证；
- SSCHA 传播并记录 `symprec` 与 `symmetry_tolerance`；
- Fourier term 循环已经展平并有 Numba batched kernel；
- 固定星展开几何已经缓存；
- 负收益的 `qpoint_workers` 线程池已经删除；
- 采样器按代表点对角化，但保留完整随机自由度、$q/-q$ 配对和边界点实自由度。

这些内容要继续由现有直接全网格 oracle、diamond、hcp、GaAs、反幺正、非对角超胞和
大 shear 测试保护。

### 1.2 真正的合并阻断项

当前不能合并的核心原因不是极端整数输入，而是以下物理和运行时问题：

1. `LoopSCPH` 默认把 Gamma 点三个数值上略大于零的平移模送入
   $k_BT/\omega^2$，使协方差被零模舍入噪声支配；
2. SCPH 与 harmonic sampler 对 Gamma 平移子空间的处理不一致；采样器会投影，SCPH 不会；
3. FC4 收缩得到的 loop correction 没有独立的 ASR 证书；错误可能直到更新 FC2 的
   full-star 门禁才以“对称性失败”的形式暴露；
4. 动力学矩阵在给定相对容差内协变，不保证奇异谱函数
   $f(\lambda)=1/\lambda$ 产生的协方差仍在同一容差内协变；
5. NaN/Inf 可以绕过 full-star 和 Hermiticity 门禁；
6. `qpoint_workers` 已删除，但 K4As4Pt2 教程脚本仍传入它，提交的 `run.log` 不是当前
   HEAD 可重现的日志；
7. CHANGELOG 和 exceptions 页面仍把主要门禁描述成 little-group 检查。

### 1.3 不再视为现实材料的合并阻断

`exact_integer_product` 对 `np.int64(-2**63)` 的绝对值边界处理不完整。真实超胞矩阵
远达不到这个量级，因此它是底层“exact”契约的健壮性缺口，不是一般材料的物理阻断。

处理要求降级为：

- 在本文最后的独立低优先级提交中修复，或明确限制并拒绝不现实的矩阵范围；
- 不得以该极值问题阻塞 Gamma/ASR 主线；
- 若函数 docstring 继续宣称“不会静默溢出”，仍必须补 `-2**63` 的回归测试。

## 2. 物理契约：三个问题必须分开

### 2.1 几何对称性、平移零模和热力学模策略不是同一个容差

必须区分：

- `symprec`，单位 Å：由 spglib 识别结构和倒空间星；
- `symmetry_tolerance`，无量纲相对误差：验证矩阵在空间群作用下的协变性；
- `frequency_cutoff_thz`，单位 THz：用户明确选择是否排除非平移的低频物理模。

三个 Gamma 平移自由度不是“低频物理模”，不得由 `frequency_cutoff_thz` 猜测或筛选。
它们由质量加权平移子空间精确识别并结构性删除。因此默认 cutoff 可以保持为
$0$ THz，而不会重新纳入平移模。

### 2.2 质量加权 Gamma 平移子空间

设原胞有 $n$ 个原子、质量为 $m_a$。构造三个列向量：

$$
B_{(a\alpha),\beta}=\sqrt{m_a}\,\delta_{\alpha\beta},
$$

并逐列归一化，使：

$$
B^\dagger B=I_3.
$$

内部位移空间投影为：

$$
Q=I-BB^\dagger.
$$

禁止通过“选择与 $B$ 重叠最大的三个本征向量”决定要删除的模。软光学模与平移模接近时，
该规则依赖任意本征矢规范。实现必须先构造 $B$ 的确定性正交补 $C$：

$$
C^\dagger C=I_{3n-3},\qquad C^\dagger B=0,
$$

然后只对内部矩阵对角化：

$$
D_{\mathrm{int}}(\Gamma)=C^\dagger D(\Gamma)C.
$$

Gamma 协方差从内部空间 lift 回完整质量加权空间：

$$
W(\Gamma)=C\,f_T(D_{\mathrm{int}})\,C^\dagger.
$$

这样三个平移自由度从未进入 $f_T$，而不是先产生巨大方差再置零。

### 2.3 非平移软模必须是显式物理策略

对 $D_{\mathrm{int}}(\Gamma)$ 和所有非 Gamma 点：

- 正本征值按 classical 或 quantum 统计构造方差；
- 精确或数值非正的内部模不得继续通过 `sqrt(abs(lambda))` 静默变成稳定模；
- 默认策略应失败并报告 q label、mode index、频率/本征值和当前温度；
- 若 SCPH 为探索软模显式支持 `imaginary_modes="absolute"` 或 `"exclude"`，必须由调用方明确选择，写入结果、metadata 和日志；
- `frequency_cutoff_thz > 0` 只处理非平移模，结果必须记录被排除的完整网格加权模数；
- 刚好落在 cutoff 边界的包含规则在 SCPH、sampler 和自由能路径中必须完全一致。

K4As4Pt2 若确实需要用绝对频率启动软模自洽，教程必须显式写出该选择，不得依赖默认
`abs(eigenvalue)`。

## 3. 目标架构

### 3.1 唯一的声子模热力学内核

在 `mlfcs.reciprocal` 内建立共享模块，建议：

```text
src/mlfcs/reciprocal/
├── modes.py          # 平移子空间、谱策略、协方差矩阵
├── symmetry.py       # 星作用和验证
├── statistics.py     # 标量热力学函数
├── sampling/
└── scph/
```

`modes.py` 至少提供内部对象：

```python
@dataclass(frozen=True, slots=True)
class ModePolicy:
    statistics: str
    temperature: float
    frequency_cutoff_thz: float
    imaginary_modes: str

@dataclass(frozen=True, slots=True)
class ModalCovariance:
    eigenvalues: np.ndarray
    frequencies_thz: np.ndarray
    included: np.ndarray
    translations: np.ndarray
    matrix: np.ndarray

def mass_weighted_translations(masses: np.ndarray) -> np.ndarray: ...
def internal_mode_basis(masses: np.ndarray) -> np.ndarray: ...
def modal_covariance(matrix, masses, *, is_gamma, policy) -> ModalCovariance: ...
```

名称可以调整，但必须满足：

- SCPH 与 sampler 调用同一个 Gamma 内部空间实现；
- `statistics.mode_sigma` 不再自行决定哪些模存在；它只计算已被策略接受的正模；
- `modal_covariance` 返回 $V\operatorname{diag}(\sigma^2)V^\dagger$，消费方不能再次手写
  `eigh + mode_sigma + matrix multiply`；
- 所有返回数组有限；发现 NaN/Inf 立即失败；
- 不把本征向量逐列当作跨简并子空间的物理身份。

### 3.2 与 ASR null-space 计划的关系

`asr-null-space-unification-plan.md` 定义的是实空间 IFC 参数 $θ$ 的约束：

$$
A\theta=0,\qquad \theta=Nz.
$$

本文的 $B$ 和 $C$ 定义的是 Gamma 动力学矩阵上的质量加权平移/内部子空间。两者表达同一
平移不变性，但作用在不同向量空间，不能把一个矩阵冒充另一个矩阵。

必须用桥接测试证明：

1. 从 `TranslationalNullSpace` lift 出来的 FC2 满足 $D(\Gamma)B=0$；
2. 有意破坏实空间 ASR 后，Gamma 证书失败；
3. Gamma 投影只负责结构性排除三个刚体平移，不得掩盖明显的实空间 ASR 破坏；
4. SCPH loop correction 若来自满足 FC4 ASR 的模型，应在数值精度内满足 FC2 ASR；否则
   报告 correction 的 ASR 残差，而不是把它误报成一般的 star covariance 错误。

SCPH 默认不得静默把一个明显不满足 ASR 的 correction 投影回去。正确顺序是：

1. 输入 FC2/FC4 在拟合或有限差分阶段通过统一 null space 获得物理可行性；
2. SCPH 验证 loop contraction 保持该不变式；
3. 只有舍入量级的漂移才能由共享投影器清理，并且投影前后残差都必须记录；
4. 超过允许 backward-error bound 时失败，要求修复 FC4 或收缩实现。

不得在 `reciprocal` 内再实现一套与 `constraints` 不一致的实空间 ASR 方程。

### 3.3 协方差的 full-star 证书

动力学矩阵满足近似协变并不足以证明协方差满足近似协变，因为 classical 路径含
$D^{-1}$，在小本征值附近条件数会放大误差。

每次 SCPH sweep 至少需要下面两种证书之一；第一版优先实现直接证书：

#### 方案一：直接协方差证书

1. 对代表点用共享 `modal_covariance` 构造 $W(q_s)$；
2. 用已有 `ReciprocalExpansionPlan` 展开到所有成员；
3. 对所有 full-grid 点用同一 mode policy 直接构造 $W_{\mathrm{direct}}(q)$；
4. 比较 direct 与 expanded，并报告最坏代表、成员、操作、反幺正标志、残差、尺度、
   最小 included frequency 和 mode policy。

该版本会为验证额外对角化 $N_q$ 个矩阵，但它先建立物理 oracle。不能为了保留漂亮的
对角化计数而跳过证明。

#### 方案二：经过证明的条件数证书

后续若要恢复只对角化 $N_{\mathrm{irr}}$ 个矩阵，必须给出并测试可复核的谱函数误差界，
至少包含动力学矩阵残差、最小 included spectral gap 和 cutoff 边界距离。不能只凭
“$D$ 已通过 symmetry gate”删除协方差证书。

基准必须分别报告：

- 生产协方差的代表点对角化数；
- 默认安全门禁额外对角化数；
- 显式关闭门禁后的对角化数；
- 门禁、展开、Fourier 求和和 loop contraction 各自耗时。

开发期默认选择正确性，不把验证成本伪装成生产加速。

### 3.4 非有限数失败策略

建立一个共享的内部检查入口，覆盖：

- 动力学矩阵；
- eigenvalues、frequencies、eigenvectors；
- modal covariance；
- expanded matrix；
- residual、scale、allowed；
- loop correction 与更新 FC2。

只要其中存在 NaN 或 Inf，必须在任何 `>`、`max` 或范数比较之前失败。异常要包含 context、
数组角色和可定位的 q label/iteration；不得让 `nan > allowed` 的假值静默通过。

`symmetry_tolerance=None` 只能关闭对称性数值比较，不能允许非有限物理数据继续运行。

## 4. 分阶段实施

### P0：rebase、基线和失败测试

1. structure-relation 分支合入 `dev` 后，将 `reflector` rebase 到新 `dev`；
2. 记录 rebase 前后的提交映射，不把用户 scratch 文件纳入提交；
3. 运行并保存当前聚焦与全量测试结果；
4. 在修改生产代码前增加以下失败测试：
   - 默认 cutoff 下物理 FC2 的 Gamma 平移不产生巨量协方差；
   - 软光学模不会被错误识别成三个平移模；
   - 非 ASR FC2 的 $D(\Gamma)B$ 证书失败；
   - loop correction 的 ASR 破坏被准确定位；
   - $D$ 门禁通过但 $W$ 门禁失败的近零模反例；
   - full-star 和 Hermiticity 门禁拒绝 NaN、`+inf`、`-inf`；
   - 当前 K4As4Pt2 脚本因 `qpoint_workers` 抛 `TypeError` 的回归证据。

删除或改写当前固定坏行为的测试：

```text
test_default_cutoff_covariance_is_dominated_by_the_gamma_zero_modes
```

新测试必须断言平移贡献为零且内部模协方差有限，不能再断言协方差大于 $10^8$。

### P1：共享 Gamma 内部子空间

1. 实现 `mass_weighted_translations` 和确定性 `internal_mode_basis`；
2. 测试 $B^\dagger B=I$、$C^\dagger C=I$、$B^\dagger C=0$；
3. 测试原子重排、同位素质量变化和 Cartesian 正交旋转下的 projector 协变性；
4. 将 sampler 当前私有 `_translation_basis`、`_project_gamma` 和“最大重叠三个本征向量”逻辑
   迁移到共享实现；
5. 将 SCPH Gamma 协方差迁移到同一共享实现；
6. 证明旧 sampler 的统计协方差保持一致；固定随机 seed 的逐位样本若因确定性内部基变化而
   改变，可以破坏性重生成，但统计量和 exact supercell covariance 必须保持。

### P2：统一 mode policy

1. `mode_sigma` 只接受已经验证为严格正且有限的频率或本征值；
2. SCPH 增加与 sampler 一致的显式 `imaginary_modes` 策略；
3. 默认不得静默使用 `abs(eigenvalue)`；
4. Gamma translations 永远不进入 cutoff 计数；
5. 结果对象和 metadata 记录 statistics、temperature、cutoff、imaginary policy、排除模数和
   最小 included frequency；
6. 对 cutoff 下、恰好在边界、略高于边界写测试；
7. classical、quantum、$T=0$ 三种路径都验证有限性和解析极限。

### P3：ASR 证书与 loop correction

1. 为 lattice FC2 建立共享的 Gamma acoustic residual：

   $$
   r_{\mathrm{ASR}}=\lVert D(\Gamma)B\rVert,
   $$

   同时报告相对于全网格动力学矩阵尺度的值；
2. 对 bare FC2、warm-start FC2、每轮 correction、updated FC2 和最终 FC2 分别验证；
3. correction 的错误必须明确写 `loop correction violates ASR`，包含 iteration、temperature、
   residual、scale 和最坏分量；
4. 与 `TranslationalNullSpace` 增加桥接属性测试；
5. 若舍入漂移需要投影，只能调用 constraints 包拥有的共享投影器，并记录投影前后值；
6. 投影后仍必须通过 full-star covariance 和 Hermiticity，不得假定 ASR 投影自动保持空间群；
7. 若 K4As4Pt2 的 correction 在修复 Gamma 后仍明显违反 ASR，停止并检查 FC4 ASR 与
   contraction index/translation 约定，不得通过放宽 `symmetry_tolerance` 合并。

### P4：协方差 full-star 验证

1. 增加 `require_star_covariance_matrix` 或职责等价的入口；
2. direct 和 expanded 必须使用同一 `ModePolicy` 和 Gamma 内部空间；
3. 覆盖 unitary、antiunitary、$G\ne0$ stored-label gauge、自共轭边界点；
4. 覆盖近简并正模，结果只比较协方差矩阵/谱 projector，不比较单个本征向量；
5. 覆盖近零内部模的误差放大反例；
6. 先交付 full-grid 直接 oracle，再决定是否实现条件数证书；
7. 任何优化版本都必须逐点对拍直接 oracle，误差界在测试中明确写出。

### P5：修复所有 failure gate

1. `require_star_covariance` 在计算残差前验证 direct/expanded 全部有限；
2. `require_hermitian` 在比较前验证 matrix、scale、residual、allowed 全部有限；
3. sampler `_validate_expansion`、SCPH updated FC2 gate 和 frequency path 使用同一 finite helper；
4. `symmetry_tolerance=None` 不绕过 finite 检查；
5. 对空数组、零尺度和全零合法模型保留明确定义；
6. 补 NaN/Inf 回归测试，并故意把 NaN 放在非最坏成员，证明不会被 `max` 顺序掩盖。

### P6：K4As4Pt2 端到端物理验收

1. 从教程脚本删除 `qpoint_workers=4`；
2. 明确选择 soft-mode policy；若教程需要 `imaginary_modes="absolute"`，在脚本和文档解释原因；
3. 从当前 HEAD 覆盖运行 `run.py`，不得复制旧日志；
4. `run.log` 必须包含完整 stdout、stderr、traceback，并被 Git 跟踪；
5. 不能把失败日志当作成功结果合并；允许未收敛只限算法达到 iteration budget 且返回有限、
   满足 symmetry/ASR 证书的最终迭代，并必须在图和文档中标明 `converged=False`；
6. 报告每轮最小内部频率、排除模数、ASR residual、star residual、correction norm 和收敛量；
7. 重新生成 SCPH 图并目视检查软模、路径标签、单位、图例和无裁切；
8. 若修复后轨迹与旧日志不同，报告物理原因，不能以“末位漂移”概括。

### P7：文档和 API 清理

更新中英文：

- SCPH theory/API；
- SSCHA theory/API；
- exceptions；
- units and parameters；
- CHANGELOG。

必须说明：

1. 主门禁是 full-star，不是 little group；
2. `symprec` 是 Å 制几何精度；`symmetry_tolerance` 是相对矩阵验证精度；
3. Gamma translations 由质量加权 null space 结构性排除；
4. cutoff 不负责识别三个平移模；
5. imaginary-mode policy 的默认行为和显式选项；
6. 协方差门禁为何比动力学矩阵门禁更强；
7. scale 的真实定义是矩阵无穷范数还是最大元素，文档必须与代码一致；
8. `qpoint_workers` 已直接删除，不提供兼容别名；
9. `symmetry_tolerance=None` 关闭哪些比较、哪些 finite/shape 检查永远不能关闭。

所有 Markdown 数学只能使用 `$...$` 和 `$$...$$`。

### P8：正确性完成后的性能工作

物理验收通过后再完成原计划遗留的两个热点。

#### P8.1 ragged covariance kernel

把 `(a, b, r)` dict 热循环预编译为：

```text
keys          int32   (n_keys, 5)
block_first   int32   (n_keys,)
block_second  int32   (n_keys,)
translations int32   (n_keys, 3)
accumulator   complex128 (n_keys, 3, 3)
```

要求：

- `prange` 放在独立 star/member 或 key 维，不能包住 LAPACK eigensolver；
- 不在 nopython 内使用 tuple-key dict、dataclass 或 `np.einsum`；
- 与 Python oracle 对拍每个 key，而不只比较总 norm；
- serial、不同 Numba thread 数和 `NUMBA_DISABLE_JIT=1` 结果一致；
- warm benchmark 单独报告，不把首次编译计入或藏掉。

#### P8.2 sampler field synthesis

采样器的 RNG 和字段合成分开：

- NumPy `SeedSequence/default_rng` 继续在 Python 层产生固定 coefficients；
- Numba 只合成相位、实空间场和 atom scatter；
- 不为进入 nopython 改随机流；
- 固定 seed 样本若 P1 已因物理修复重生成，P8 不得再次改变；
- 理论协方差、经验协方差和 exact supercell covariance 三方继续对拍。

### P9：低优先级整数健壮性

单独提交修复 `exact_integer_product` 的 signed minimum：

- 在取绝对值前转 Python `int`，或显式识别 signed dtype 最小值；
- fast path 的 bound 必须在 Python integer 中计算；
- 测试 `-2**63` 乘 $2$、乘 $-1$ 和混合符号矩阵；
- 若选择限制输入范围，错误必须先于 NumPy 乘法发生并说明允许范围；
- 该提交不得与 Gamma/ASR 物理修复混在一起。

## 5. 测试矩阵

### 5.1 单元与属性测试

至少覆盖：

- 单原子 cubic、多原子 diamond、hcp、zincblende GaAs；
- 对角、非对角和 shear 超胞；
- Gamma、一般 $q/-q$ 对、自共轭边界点；
- unitary 与 antiunitary 星成员；
- stored-label reduction 中 $G\ne0$ 的成员；
- 原子重排、质量变化和 Cartesian 正交旋转；
- 恰好三个平移自由度；
- 平移附近存在软光学模；
- 正常正定、真实虚频、零内部模、cutoff 边界；
- NaN/Inf 出现在 direct、expanded、scale、correction 的不同位置；
- ASR 合法/非法 FC2 与 FC4；
- NumPy oracle、Numba、禁用 JIT 三条路径。

### 5.2 必须保持的计数不变量

对每个 grid 断言：

$$
\sum_s w_s=N_q,
$$

并分别报告：

- $N_q$；
- $N_{\mathrm{irr}}$；
- 生产本征求解数；
- 验证本征求解数；
- Gamma 删除的平移自由度固定为 3；
- 非平移 cutoff/imaginary policy 排除的加权模数。

不能把“验证也做了 $N_q$ 次对角化”写成“算法只做 $N_{\mathrm{irr}}$ 次”。生产计算和安全
证书必须分栏报告。

### 5.3 物理 oracle

必须同时具备：

1. 直接 full-grid Fourier covariance；
2. irreducible-star covariance；
3. exact finite-supercell Cartesian covariance；
4. sampler 理论 covariance；
5. 足够样本数的经验 covariance；
6. real-space ASR 与 Gamma $D(\Gamma)B$ 桥接。

比较对象是矩阵、block、谱 projector 和统计量；简并子空间中不得逐列比较任意本征向量。

## 6. 验收命令

实现者应按 rebase 后实际测试文件调整路径，但至少给出：

```bash
pytest -q \
  tests/test_reciprocal_diagnostics.py \
  tests/test_reciprocal_covariance_path.py \
  tests/test_reciprocal_sampling_path.py \
  tests/test_reciprocal_full_grid_oracle.py \
  tests/test_reciprocal_frequency_path.py \
  tests/test_anharmonic_scph.py

pytest -q tests/test_architecture_dependencies.py
pytest -q
ruff check src tests scripts
python scripts/check_docs.py
mkdocs build --strict
python scripts/sync_readmes.py --check

rg -n "qpoint_workers|mlfcs\.phonon" src tests tutorial docs
rg -n "little group|little-group" CHANGELOG.md CHANGELOG_ZH.md docs
rg -n "nan|inf" tests/test_reciprocal_diagnostics.py
```

搜索结果不是一律要求为空：little group 可以作为局部诊断出现，但所有命中必须人工分类；
教程生产脚本不得再出现 `qpoint_workers` 或旧 `mlfcs.phonon.*` 路径。

## 7. 建议提交拆分

每个提交必须能独立审查，建议顺序：

1. `test: expose Gamma zero-mode and nonfinite reciprocal failures`
2. `reciprocal: share the mass-weighted Gamma internal subspace`
3. `reciprocal: make soft-mode thermodynamics an explicit policy`
4. `scph: certify ASR of every loop correction and iterate`
5. `scph: validate covariance on every expanded star member`
6. `reciprocal: reject nonfinite matrices before tolerance gates`
7. `tutorial: rerun K4As4Pt2 with the repaired modal physics`
8. `docs: describe the Gamma null space and full-star covariance gate`
9. `performance: compile the ragged covariance assembly`
10. `performance: compile sampler field synthesis without changing RNG`
11. `integer-lattice: harden the exact product signed boundary`

教程日志、核心实现、性能优化和文档不得压成一个提交。P9 可以在倒空间合并后单独完成；若
保留“exact never wraps”的公开承诺，则必须在发布前完成。

## 8. 交付报告

最终报告必须包含：

1. 新 `dev` 基线、rebase 后 HEAD 和完整提交列表；
2. Gamma 平移子空间的维数、正交残差和投影残差；
3. 修复前后默认 cutoff 的 Gamma covariance 最大 block；
4. bare、correction、updated、final FC2 的最大 ASR residual；
5. 动力学矩阵 full-star residual 与 covariance full-star residual；
6. 每个测试材料的 $N_q$、$N_{\mathrm{irr}}$、生产/验证对角化数；
7. K4As4Pt2 的 mode policy、收敛状态、每轮残差和全部产物差异；
8. NaN/Inf 反例的实际拒绝消息；
9. P8 前后分项 benchmark、JIT cold/warm 时间、线程数和工具链版本；
10. 全套测试、ruff、双语文档、严格构建和 README 同步结果；
11. tracked/untracked 状态及所有未纳入提交的用户文件。

## 9. 完成定义

只有同时满足以下条件才可合并：

1. SCPH 和 sampler 使用同一质量加权 Gamma 内部子空间；
2. 默认 cutoff 不再把三个平移模送入 $1/\omega^2$；
3. 非平移 imaginary/soft-mode 行为是显式、记录且测试过的策略；
4. loop correction 和所有 FC2 iterate 都有可定位的 ASR 证书；
5. covariance 本身通过 full-star 证书，不能只证明动力学矩阵；
6. NaN/Inf 无法绕过任何默认 failure gate；
7. K4As4Pt2 脚本在当前 HEAD 可运行，日志不是陈旧产物；
8. CHANGELOG、exceptions 和理论/API 页面与 full-star 实现一致；
9. 直接 full-grid、irreducible、finite-supercell 和 sampler covariance oracle 一致；
10. 全套测试与文档门禁通过，架构隔离仍成立。

P8 的两个性能阶段可以在物理合并后继续，但若延期，发布说明必须如实报告当前验证成本；
不得把未实现的 4–8 倍预估写成实测收益。P9 的极端整数健壮性不再阻止一般材料的物理合并。

## 10. 明确不在本计划内

- 不重新实现 spglib；
- 不把 `symprec` 用作频率、矩阵或 ASR 容差；
- 不用提高 `frequency_cutoff_thz` 掩盖 Gamma 平移零模；
- 不用放宽 `symmetry_tolerance` 掩盖 K4As4Pt2 correction 错误；
- 不恢复 `qpoint_workers`；
- 不改变已经验证的 star/gauge/antiunitary 定义；
- 不在 Numba 内重写随机数流；
- 不让主线包反向依赖 `mlfcs.reciprocal`；
- 不在本轮重复 structure-relation 的单一 `symprec` 重构；
- 不把一般材料不可能达到的整数极值与真实声子物理问题置于同一优先级。
