---
title: 倒空间对称性约化与展开重构计划
audience:
  - advanced
  - developer
status: research
code_verified: 4.0.0a6
localized_only: true
---

# 倒空间对称性约化与展开重构计划

## 1. 目标与结论

本重构的目标不是简单删除一部分 $q$ 点，而是建立一套完整、精确且可验证的倒空间对称性模型：

1. 用整数标签构造与给定超胞网格兼容的空间群作用；
2. 将完整 $q$ 网格严格分解为不可约代表点及其星；
3. 只在不可约代表点上构造和对角化动力学矩阵；
4. 按物理量的变换规律，将标量、矩阵、协方差和随机模展开回完整网格；
5. 对需要完整布里渊区求和的计算使用星成员或严格等价的权重，不能把“约化计算量”误写成“删除物理自由度”。

当前实现只对谐波采样中的 $q$ 与 $-q$ 做了实场配对；SCPH 仍在完整商群网格上逐点对角化。这不是空间群意义下的不可约布里渊区约化。

预期的主收益是把动力学矩阵对角化次数从 $N_q$ 降到 $N_{\mathrm{irr}}$。理想加速比约为 $N_q/N_{\mathrm{irr}}$，上限受网格保持子群阶数限制。实空间协方差求和、完整随机场生成和最终展开仍至少具有 $O(N_q)$ 工作量，因此不得宣称整个算法获得同等倍数的端到端加速。

## 2. 影响范围判断

### 2.1 直接影响

现有以下模块需要修改，随后收缩到第 3 节定义的单一倒空间包：

- `src/mlfcs/structure/reciprocal.py`：完整商群网格、精确群作用、星分解和不可约网格数据模型；
- `src/mlfcs/phonon/reciprocal.py`：声子空间的表示矩阵、Fourier gauge、物理量展开；
- `src/mlfcs/phonon/scph/fourier.py`：不可约点上的动力学矩阵和频率计算；
- `src/mlfcs/phonon/scph/solver.py`：SCPH 频率、收敛范数、协方差求和和并行任务划分；
- `src/mlfcs/phonon/sampling/harmonic.py`：复用不可约点本征分解并恢复完整的随机自由度；
- `src/mlfcs/phonon/sscha/solver.py`：原则上只需适配谐波采样器的新结果和诊断字段；
- 对应的导出、测试、理论文档和 API 文档。

### 2.2 不只影响 SCPH 和 SSCHA

用户判断在主计算链上基本正确：力常数拟合、轨道生成和力常数实空间展开不依赖这里的 $q$ 网格，不应因本重构改变结果。

但现有公开接口中还有三个直接消费者：

- `mlfcs.phonon.harmonic_frequencies`；
- `mlfcs.phonon.HarmonicSampler`；
- `perturb_structures(method="harmonic")`。

它们必须随新模型一起定义清楚。`perturb_structures(method="gaussian")` 不受影响。

### 2.3 明确不应改动的主线

除非验证暴露既有物理错误，本计划不修改：

- 候选簇、轨道代数、拟合器和线性求解器；
- 有限差分和力常数实现；
- ASR、力常数 IO 和外部格式导出；
- 计算器接口；
- 普通能带路径生成和绘图；
- 由外部程序自行完成网格约化的热输运流程。

重构必须以“上述模块的测试和公共结果无变化”作为隔离性证据，而不是顺手调整它们。

## 3. 强制包边界与目录收缩

### 3.1 目标目录

倒空间网格、谐波模、谐波采样、SCPH 和 SSCHA 必须收缩到一个独立包。建议目标为 `src/mlfcs/reciprocal/`：

```text
src/mlfcs/reciprocal/
├── __init__.py
├── grid.py
├── symmetry.py
├── fourier.py
├── modes.py
├── statistics.py
├── temperature.py
├── sampling/
│   ├── __init__.py
│   └── harmonic.py
├── scph/
│   ├── __init__.py
│   ├── fourier.py
│   └── solver.py
└── sscha/
    ├── __init__.py
    └── solver.py
```

文件可在实现中进一步拆分，但不能重新散落到 `structure`、`phonon` 或顶层模块。这里的“一个文件夹”指一个有明确公共入口和单向依赖规则的 Python 包，而不是用旧路径包装新实现。

现有文件按下列规则迁移：

| 现有位置 | 目标位置或处置 |
| --- | --- |
| `structure/reciprocal.py` | `reciprocal/grid.py` |
| `phonon/reciprocal.py` | 删除转发层；实现进入 `reciprocal/grid.py` 和 `reciprocal/symmetry.py` |
| `phonon/modes.py` | `reciprocal/modes.py` |
| `phonon/statistics.py` | `reciprocal/statistics.py` |
| `phonon/temperature.py` | `reciprocal/temperature.py` |
| `phonon/scph/` | `reciprocal/scph/` |
| `phonon/sscha/` | `reciprocal/sscha/` |
| `phonon/sampling/harmonic.py` | `reciprocal/sampling/harmonic.py` |

纯笛卡尔 Gaussian 扰动本身不属于倒空间。应将它拆成外部的低层工具，例如 `mlfcs.sampling.gaussian`，允许倒空间包依赖它。现有同时分派 Gaussian 和 harmonic 的 `perturb_structures` 必须拆开或移入倒空间包，不能留在外部再反向导入 `reciprocal`。

### 3.2 单向依赖规则

倒空间包是依赖图最外层的功能包，没有其他生产包以它为依赖。允许的方向为

$$
\texttt{mlfcs.reciprocal}
\longrightarrow
\{\texttt{structure},\texttt{force\_constants},\texttt{sampling.gaussian},\texttt{io 基础类型}\}.
$$

反方向一律禁止：

$$
\{\texttt{structure},\texttt{force\_constants},\texttt{calculators},\texttt{fitting},\texttt{interaction},\texttt{io}\}
\not\longrightarrow
\texttt{mlfcs.reciprocal}.
$$

具体约束如下：

- `structure` 只提供晶体、整数晶格、空间群操作和超胞映射等基础数据，不知道 $q$ 网格；
- `force_constants` 只提供实空间力常数及结构关系，不调用 Fourier、SCPH 或 SSCHA；
- `reciprocal` 可以读取上述基础对象并在其上建立倒空间模型；
- SCPH、SSCHA 和 harmonic sampling 之间允许在 `reciprocal` 包内依赖共享内核，但不得通过顶层公共 API 互相绕行；
- 仓库根 `mlfcs/__init__.py` 不再导出 `LoopSCPH`、`SSCHA`、`HarmonicSampler` 或 harmonic `perturb_structures`，用户必须从 `mlfcs.reciprocal` 的明确入口导入；
- 不保留 `mlfcs.phonon.*` 或 `mlfcs.structure.reciprocal` 的兼容转发模块。

最后两条是有意的破坏性 API 变更，用于保证“外部不依赖倒空间包”在源码依赖图中真实成立，而不只是移动文件后保留一圈反向 import。

### 3.3 公共入口与内部入口

`mlfcs.reciprocal.__init__` 只导出稳定的领域对象，例如不可约网格结果、`harmonic_frequencies`、`HarmonicSampler`、`LoopSCPH` 和 `SSCHA`。`_dynamical`、星展开内核、协方差块和随机模构造保持包内私有。

包内代码应使用相对导入或从具体子模块导入，禁止从 `mlfcs.reciprocal` 顶层回导自身对象，以免形成初始化环。SSCHA 只能依赖公共的 harmonic sampling 内核，SCPH 和 SSCHA 不能互相依赖。

### 3.4 架构测试

在 `tests/test_public_api_contract.py` 或独立架构测试中增加硬性检查：

1. 扫描 `src/mlfcs/reciprocal` 之外的 Python 文件，禁止出现 `mlfcs.reciprocal` 导入；测试和文档示例不属于 `src` 扫描范围，不为任何生产模块设置白名单；
2. 扫描倒空间包，禁止依赖 `fitting`、`calculators` 或其他高层业务包；
3. 验证 `structure`、`force_constants`、`interaction` 和 `io` 的导入闭包不包含 `mlfcs.reciprocal`；
4. 验证删除旧 `phonon` 路径后，不存在兼容 shim、动态 `sys.modules` 别名或延迟反向导入；
5. 单独导入每个公共子包，捕获循环依赖和依赖可选组件的问题。

该架构测试是合并门槛。只要外部生产模块仍导入倒空间包，目录收缩就不算完成。

## 4. 当前实现及主要问题

`reciprocal_quotient_grid(S)` 已经正确枚举超胞平移有限群的全部特征标。令超胞整数矩阵为 $S$，$D=|\det S|$，当前网格用整数标签 $\ell$ 表示

$$
q = \frac{\ell}{D} \pmod{\mathbb{Z}^3},
$$

并验证

$$
qS^{\mathsf T}\in\mathbb{Z}^3.
$$

这部分应保留为完整网格的唯一基础，不应用浮点坐标去重或匹配。

当前浪费主要来自：

- `LoopSCPH._frequencies` 在每轮迭代对全部 $q$ 点对角化；
- `LoopSCPH._covariance` 对全部 $q$ 点批量构造并对角化；
- `harmonic_frequencies` 对全部 $q$ 点对角化；
- `HarmonicSampler._prepare_modes` 只利用 $D(-q)=D(q)^*$，没有利用空间群的 $q$ 星。

另一个结构性问题是 Fourier gauge 尚未显式化：

- SCPH 使用相位 $\exp[2\pi i q\cdot(R+\tau_b-\tau_a)]$；
- 谐波采样的动力学矩阵使用 $\exp(2\pi i q\cdot R)$。

二者本征值相同，但本征矢相差逐原子相位。若直接共享对称展开矩阵，多原子晶体和非对称型空间群会产生错误结果。因此必须先选择统一规范，或让 gauge 成为数据模型中的显式字段。

## 5. 不可妥协的物理与数学约束

### 5.1 只使用保持当前网格的对称操作

设原胞分数坐标采用行向量，空间群操作为

$$
x' = xR^{\mathsf T}+t.
$$

则倒空间行向量变换为

$$
q' = qR^{-1}.
$$

不是每个原胞点群操作都保持任意超胞网格。例如各向异性网格通常不允许交换不同密度的轴。必须精确筛选网格保持子群 $G_S$，不能先用完整点群生成星，再用浮点容差把落在网格外的点丢弃。

可复用 `SymmetryOperations.from_primitive_operations` 中已有的整数兼容条件。实现时应将这一条件提取成共享的结构层函数，并证明它与倒空间标签闭包等价。对每个保留操作，还必须逐标签验证变换结果仍属于完整网格。

### 5.2 标签作用必须精确

对于分母为 $D$ 的标签，候选作用为

$$
\ell' = \ell R^{-1} \pmod D.
$$

$R$ 是行列式为 $\pm1$ 的整数矩阵，因此 $R^{-1}$ 仍为整数矩阵。最终公式必须由本项目的行向量约定和 Fourier 相位推导，并由非对角超胞测试固定；不得靠尝试转置直到测试通过。

### 5.3 时间反演是反幺正操作

对实力常数有

$$
D(-q)=D(q)^*.
$$

时间反演应作为可显式启用的反幺正星操作，使用现有精确负标签，而不是伪装成普通空间旋转。空间群星与 $q/-q$ 配对必须一次性构造，避免重复计权。第一版仅支持无磁、实力常数模型；将来若支持破坏时间反演的模型，必须能关闭该操作。

### 5.4 标量权重不能替代张量展开

频率、自由能等标量可按星权重求和。实空间协方差包含原子块、笛卡尔分量和相位，不能只把代表点贡献乘以星大小。必须将代表点上的协方差矩阵按每个星成员的表示展开，再执行带相位的 Fourier 求和，或推导并验证严格等价的星求和算子。

### 5.5 对称性不会删除随机自由度

对称等价的两个不同 $q$ 点在谐波系综中仍有独立的随机振幅，只有 $q$ 与 $-q$ 受到实位移条件的共轭约束。禁止为一个空间群星只抽取一组随机数并将其旋转到所有成员；那会错误地给不同模引入相关性。

正确做法是复用代表点的本征分解或协方差平方根，但为不同完整网格成员抽取独立随机变量，并对负点施加共轭关系。

### 5.6 简并空间按投影算子处理

简并本征矢没有唯一方向。不能要求逐列本征矢在展开前后相等，也不能依赖本征矢排序来定义物理映射。测试和实现应优先比较：

- 排序后的本征值；
- 简并子空间投影算子；
- 加权协方差 $Vf(\Omega)V^\dagger$；
- 动力学矩阵的协变关系。

## 6. 建议的数据模型

### 6.1 精确星分解

在 `reciprocal/grid.py` 中保留 `ReciprocalQuotientGrid`，并新增只依赖整数晶格和空间群基础数据的数据类：

```python
@dataclass(frozen=True, slots=True)
class ReciprocalGridSymmetry:
    operation_indices: np.ndarray
    rotations: np.ndarray
    label_permutations: np.ndarray

@dataclass(frozen=True, slots=True)
class ReciprocalStar:
    representative: int
    members: np.ndarray
    operations: np.ndarray
    antiunitary: np.ndarray

@dataclass(frozen=True, slots=True)
class IrreducibleReciprocalGrid:
    full: ReciprocalQuotientGrid
    stars: tuple[ReciprocalStar, ...]
    representatives: np.ndarray
    full_to_irreducible: np.ndarray
    full_operations: np.ndarray
    full_antiunitary: np.ndarray
    weights: np.ndarray
    little_groups: tuple[np.ndarray, ...]
```

字段名可以在实现时调整，但必须表达以下信息：

- 完整网格的稳定顺序；
- 每个不可约代表点；
- 代表点到每个完整成员所用的具体操作；
- 该映射是否包含时间反演；
- 每个完整点属于哪个星；
- 每个代表点的 little group；
- 星权重。

代表点应取其轨道中完整网格索引最小者，成员按完整网格索引排序。这样结果不依赖 spglib 返回操作的偶然顺序。若多个操作映射到同一成员，应选择稳定的最小操作索引，但保留 little group 的全部操作。

构造后必须验证：

$$
\bigsqcup_s \operatorname{star}(q_s)=Q,\qquad
\sum_s w_s=N_q.
$$

### 6.2 声子表示与 Fourier gauge

`reciprocal/grid.py` 只负责整数标签的几何作用。`reciprocal/symmetry.py` 负责由以下数据构造质量加权位移空间中的幺正表示 $U_g(q)$：

- `PrimitiveSymmetryOperations.site_permutations`；
- `PrimitiveSymmetryOperations.site_shifts`；
- `PrimitiveSymmetryOperations.cartesian_rotations`；
- 操作平移 $t$；
- 明确的 Fourier gauge。

建议全项目统一为含原子位置的 gauge，即动力学矩阵使用

$$
D_{ab}(q)=\frac{1}{\sqrt{m_am_b}}\sum_R
\Phi_{ab}(R)\exp\left[2\pi iq\cdot(R+\tau_b-\tau_a)\right].
$$

这个选择与当前 SCPH 一致。谐波采样的场重建相位需相应包含 $R+\tau_a$。若不在本次统一，则必须实现具名的 gauge 转换矩阵，禁止让两个模块各自隐式约定。

空间群展开的中心恒等式是

$$
D(gq)=U_g(q)D(q)U_g(q)^\dagger.
$$

时间反演成员还需先做复共轭。平移相位、原子置换方向和笛卡尔旋转方向必须从位移 Bloch 和推导，随后用直接重新构造的 $D(gq)$ 作为 oracle 固定下来。

little group 为

$$
G_q=\{g\in G_S\mid gq=q+G\}.
$$

默认行为应检查 $D(q)$ 的 little-group 协变残差并在超差时报告，而不是静默平均掉输入力常数的对称性错误。可提供显式的诊断或投影函数，但不能默认掩盖上游问题。

## 7. 各消费者的重构方案

### 7.1 `harmonic_frequencies`

只对不可约代表点构造和对角化动力学矩阵，然后按星映射展开排序后的频率。建议破坏现有二元元组 API，返回具名结果：

```python
HarmonicMeshResult(
    irreducible_qpoints=...,
    irreducible_frequencies=...,
    weights=...,
    grid=...,
)
```

完整数组只能通过显式的 `expand_frequencies()` 得到。这样调用方不会误以为 `qpoints` 究竟指完整网格还是不可约网格。项目仍在开发阶段，不保留旧二元元组兼容层。

### 7.2 SCPH 频率与收敛

每轮只计算不可约频率。收敛范数直接用星权重计算：

$$
\Delta\omega =
\sqrt{\frac{1}{N_qN_b}\sum_s w_s
\lVert\omega_s^{(n)}-\omega_s^{(n-1)}\rVert_2^2}.
$$

它必须与完整展开后再取均方根逐位等价。`qpoint_workers` 改为调度不可约代表点，不得在工作线程中重复展开完整网格。

`LoopSCPHResult` 应保存不可约点、频率、权重和网格映射。若用户需要完整数组，调用显式展开方法。日志增加 $N_q$、$N_{\mathrm{irr}}$、约化比和实际对角化次数。

### 7.3 SCPH 协方差

正确性优先的第一版流程为：

1. 在每个不可约代表点上计算 $D(q_s)$；
2. 对角化并形成基底无关的加权矩阵

$$
W(q_s)=V_s\,\operatorname{diag}(\sigma_s^2)\,V_s^\dagger;
$$

3. 对星中每个成员，用 $U_g$ 和必要的复共轭得到 $W(gq_s)$；
4. 从 $W(gq_s)$ 取所需原子块并乘正确 Fourier 相位；
5. 对完整星求和，最后除以 $N_q$。

这能减少昂贵的对角化，但保留完整的物理求和。第二阶段可将第 3 至 4 步融合成星求和算子以减少矩阵复制；融合实现必须以第一版为 oracle。

### 7.4 谐波采样与 SSCHA

采样器按以下层次工作：

1. 每个不可约代表点只对角化一次；
2. 用表示矩阵把本征子空间或协方差平方根展开到每个完整 $q$ 点；
3. 对不同的完整 $q$ 点抽取独立随机系数；
4. 对 $q/-q$ 对只抽取一侧，并令另一侧为共轭；
5. 对满足 $q=-q+G$ 的边界点使用实自由度；
6. 合成实空间位移并沿用现有质量归一化、虚频策略和位移截断。

为使固定种子不依赖星遍历顺序，随机流应由“全局种子 + 完整网格精确标签”派生。若无法保持旧实现逐样本相同，必须保证统计分布一致，并在破坏性变更中明确记录；不能用一次全局顺序抽样让 spglib 操作顺序影响结果。

自由能、虚频数量、总模数和排除模数可直接按星权重统计。`SamplingState` 应同时报告完整 $q$ 点数和不可约 $q$ 点数。`HarmonicSampler` 不再提供含义含混的单一 `.qpoints`，改为 `.irreducible_qpoints`、`.weights` 和显式完整网格访问。

SSCHA 不应自行实现另一套倒空间对称逻辑；它只消费重构后的 `HarmonicSampler`。

## 8. 分阶段提交计划

### 0. 先建立包边界

- 新建 `mlfcs.reciprocal` 目录和最小公共入口；
- 将现有 reciprocal、harmonic sampling、SCPH、SSCHA、statistics 和 temperature 模块机械迁移，暂不改变数值逻辑；
- 拆出不属于倒空间的纯 Gaussian 低层工具；
- 一次性更新内部调用、测试和文档导入路径；
- 删除旧 `mlfcs.phonon`、`structure/reciprocal.py` 及所有兼容转发；
- 加入单向依赖架构测试并使其通过。

该提交只改变所有权和导入路径。必须先证明机械迁移前后测试结果相同，再开始数学重构，避免把目录迁移错误与对称约化错误混在一起。

### A. 固定完整网格 oracle 和性能基线

- 为当前全网格频率、SCPH 协方差、自由能和采样协方差增加刻画测试；
- 覆盖对角、非对角、各向异性超胞以及多原子原胞；
- 记录每条路径的动力学矩阵构造数、对角化数、峰值批大小和耗时；
- 测试必须先在未重构源码上通过。

### B. 实现精确网格保持子群和星分解

- 提取超胞兼容操作筛选；
- 实现整数标签上的群作用；
- 加入时间反演的反幺正标记；
- 构造稳定星、权重、反向映射和 little group；
- 此阶段不改任何生产消费者。

### C. 统一 Fourier gauge 并实现声子表示

- 明确并统一动力学矩阵 Fourier 规范；
- 构造 $U_g(q)$ 和 gauge 转换；
- 验证动力学矩阵、协方差和简并子空间的协变性；
- 对破坏对称性的力常数提供可定位的残差诊断。

### D. 迁移频率路径

- 先迁移 `harmonic_frequencies`；
- 再迁移 SCPH `_frequencies`、收敛范数和 worker 调度；
- 修改结果类型为不可约数据加显式展开，不提供旧 API 兼容分支；
- 与阶段 A 的完整网格 oracle 比较。

### E. 迁移 SCPH 协方差

- 实现代表点 $W(q)$ 计算和逐星成员展开；
- 对每个 FC4 实际需要的 $(a,b,R)$ 协方差块逐项比较；
- 正确后再做融合和内存优化；
- 禁止以星权重简单乘代表点矩阵。

### F. 迁移谐波采样与 SSCHA

- 分离“不可约本征分解缓存”和“完整随机自由度生成”；
- 统一 $q/-q$ 配对与空间群星；
- 更新自由能、状态、随机种子和公开属性；
- SSCHA 只做新采样接口适配。

### G. 文档、基准和清理

- 更新中英文 SCPH、SSCHA 理论页和 API 页；
- 文档解释不可约计算与完整随机自由度的区别；
- 删除开发期的全网格切换开关和兼容层；
- 输出代表性体系的 $N_q/N_{\mathrm{irr}}$、对角化次数、耗时和峰值内存对比。

每一阶段应独立提交。阶段 0、B 和 C 在进入生产路径前必须可单独审阅；不要把包迁移、数学基础、消费者迁移和教程产物混在一个提交中。

## 9. 验证矩阵

### 9.1 精确代数

- 单位、复合和逆操作在标签上的作用闭合；
- 星构成完整且不相交的分区；
- 权重和等于 $N_q$；
- 非对角超胞不使用浮点匹配；
- 各向异性网格只保留真正兼容的子群；
- 乘数网格保持相同的兼容性判断；
- 时间反演开启和关闭均有明确结果。

### 9.2 晶体覆盖

至少覆盖：

- 单原子立方晶体；
- fcc 原胞；
- 六方晶体；
- 倾斜或菱方晶胞；
- 多原子原胞；
- 含非对称型操作的结构；
- 一般非对角超胞矩阵；
- 轴密度不同、会降低网格保持子群的超胞。

### 9.3 特殊 $q$ 点

- $\Gamma$ 点；
- 满足 $q=-q+G$ 的区界点；
- 一般 $q/-q$ 对；
- 非平凡 little group 上的简并点；
- 星大小小于一般位置的高对称点。

### 9.4 物理等价

- 不可约计算后展开的完整频率与全网格 oracle 相同；
- 动力学矩阵满足空间群协变和时间反演关系；
- SCPH 每个所需实空间协方差块与全网格求和相同；
- SCPH 每轮修正、收敛历史和最终 FC2 在容差内相同；
- 谐波自由能相同；
- 虚频、cutoff 和总模数的加权计数相同；
- 采样均值为零，二阶协方差与理论值和旧全网格统计一致；
- 不同星成员的随机振幅没有非物理相关；
- 简并情况比较投影或协方差，不比较任意本征矢列。

数值容差应按量的尺度分别定义。整数标签、分区、权重和映射必须逐位相同，不能使用 `allclose`。

### 9.5 隔离性

运行拟合、有限差分、轨道、ASR、IO 和计算器测试，证明未受影响。还必须用导入闭包证明这些模块不依赖 `mlfcs.reciprocal`。若这些路径出现数值变化或新增倒空间依赖，应视为越界或暴露出的独立缺陷，不得未经说明并入本重构。

## 10. 性能验收

基准至少报告：

- $N_q$、$N_{\mathrm{irr}}$ 和各星权重分布；
- 每轮动力学矩阵构造和对角化次数；
- SCPH 频率路径与协方差路径分别耗时；
- HarmonicSampler 初始化、单批采样和自由能耗时；
- 峰值内存；
- 单 worker 和多 worker 结果一致性。

合并门槛为：所有代表性高对称体系的对角化次数严格等于 $N_{\mathrm{irr}}$，完整网格展开中不得再次调用 eigensolver。低对称体系可能几乎没有加速，这是正确结果，不能通过错误扩大对称群来制造性能数据。

## 11. 失败策略与诊断

以下情况必须报错并包含矩阵、操作索引或标签等定位信息：

- spglib 无法确定原胞对称性；
- 操作不保持网格却被请求用于展开；
- 标签作用未闭合；
- 原子置换或质量与空间群操作不一致；
- little-group 协变残差超过显式容差；
- 展开映射不能覆盖完整网格或产生重复归属；
- 展开后的矩阵不满足 Hermitian 条件；
- 实位移合成后残余虚部超过容差。

`symprec` 和物理协变检查容差必须由公开构造器显式接收并记录在结果元数据中，不能藏在模块常量里。几何识别容差和动力学矩阵数值容差是两种不同概念，不能共用一个参数。

## 12. 合并检查清单

只有同时满足以下条件才能合并：

- 完整网格仍由 `ReciprocalQuotientGrid` 的精确标签唯一生成；
- 网格保持子群通过整数条件筛选；
- 不可约星、little group、时间反演和权重的数据模型完整；
- Fourier gauge 已统一或显式建模；
- 空间群表示包含原子置换、笛卡尔旋转及所需相位；
- 频率、协方差和采样分别使用适合自身物理性质的展开；
- SCPH 与谐波采样不再在完整网格重复对角化；
- 谐波采样仍保留完整数量的独立随机自由度；
- 简并模不依赖任意本征矢规范；
- 新旧全网格 oracle 的物理量验证通过；
- 主线拟合和实空间功能没有结果变化；
- 所有倒空间、谐波采样、SCPH 和 SSCHA 实现均位于 `mlfcs.reciprocal` 包内；
- 外部生产模块不导入 `mlfcs.reciprocal`，倒空间包只向外依赖基础层；
- 旧 `mlfcs.phonon.*` 和 `mlfcs.structure.reciprocal` 路径已删除且没有兼容 shim；
- 单向依赖架构测试通过；
- 中英文文档和破坏性 API 变更记录齐全；
- 性能报告证明收益来自减少对角化，而不是减少物理求和或随机自由度。

## 13. 推荐的最终边界

最终应形成一个内部三层、对外单向依赖的 `mlfcs.reciprocal` 包：

1. `reciprocal.grid` 负责有限平移群、精确 $q$ 标签、网格保持子群和星分解；
2. `reciprocal.symmetry` 与 `reciprocal.fourier` 负责 Fourier gauge、声子空间群表示和各种物理量的展开；
3. `reciprocal.scph`、`reciprocal.sampling` 和 `reciprocal.sscha` 只负责各自算法，不再各自推导 $q$ 点对称性。

该包可以依赖 `structure`、`force_constants` 等基础层，基础层和其他主线功能不得依赖该包。这一定义下，倒空间对称性重构不会侵入力常数拟合主线，但会规范化所有谐波网格消费者。实现重点不是“返回更少的 $q$ 点”，而是同时确保包边界、不可约计算、完整展开和统计自由度正确。
