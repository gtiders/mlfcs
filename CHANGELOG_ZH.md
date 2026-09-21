# 变更记录

[English](CHANGELOG.md)

本文件记录面向用户的重要变化，版本遵循语义化版本约定。

## 4.0.0a6 — 2026-09-20

### 变化

- 顶层命名空间改为直接导入全部工作流：`ForceConstantFitter`、`LoopSCPH`、`SSCHA`、
  `perturb_structures` 的 `__getattr__` 延迟加载器已删除，`import mlfcs` 会一并加载拟合与
  有限温度栈。
- `prepare_gram()` 不再接受 `batch_size`：构型逐个流式处理，编译后的设计核从相互作用 orbit
  获取并行度，因此不再暴露构型批次开关，设计矩阵的工作集也只对应单个构型。
- 力设计矩阵与 Gram 统计改由 Numba 编译核构造，不再使用 JAX。GPU 路径、`jax_platform`
  拟合参数以及 `jax` 运行时依赖全部移除：`prepare_gram()` 对每个 IFC order 只执行一次编译核，
  Gram 矩阵由 OpenBLAS 累加，拟合栈因此可移植，也不再需要物化 XLA 分块缓冲。力常数元数据
  不再包含 `jax_platform` 字段。

### 修复

- `PeriodicGeometry` 改由 ASE 的 Minkowski 约化搜索求最小像，不再调用
  `ase.geometry.find_mic`。`find_mic` 在折叠向量短于 `0.5 * min(cell.lengths())` 时跳过约化，
  而该阈值并非 Wigner-Seitz 胞的内切半径，斜胞因此会得到非最小像。最小像长度、简并像集合
  与团簇像选择现在都与真实最小像一致；`mic()` 的调用约定保持不变。

### 新增

- `LatticeFrame`（`mlfcs.structure.lattice_frame`）记录用户 primitive 晶胞到规范 Minkowski 规约代数晶胞的
  精确整数换基：`source_cell`、`algebra_cell`、unimodular 的 `source_to_algebra`/`algebra_to_source`、
  motif 对应、分数坐标与整数平移的精确换算、旋转矩阵在两个参考系间的转换，以及唯一的 $n$ 阶张量映射
  $K_n = (A^{\mathsf T})^{\otimes n}$。等价的 unimodular 输入会规约到同一参考系。
- `mlfcs.interactions.algebra.exact` 用模素数证书判定精确秩：模 $p$ 秩不超过 $\mathbb Q$ 上的秩，因此某个
  素数上满秩即为证明；秩亏则由"已用不同素数之积超过最大子式的 Hadamard 上界"证书判定。核由 Smith 标准形
  分解构造，是**饱和**整数核，而不是逐列除以最大公约数。`RankCertificateError` 表示按需素数流无法给出证书，
  `IntegerRangeError` 表示 lattice 整数超出 `int64`，而不是让 C 扩展抛出含混的转换错误。

### 变化

- 轨道代数在规约后的 lattice（scaled）参考系中判定：spglib 旋转对任何晶胞都是整数。不变核是堆叠稳定子约束的
  精确核，维数有证书，返回的整数基会与约束精确校验；浮点 Gram、其特征值阈值与 `normalize_pivot_basis`
  全部删除。
- `PrimitiveInteractionOrbit` 用 `exact_lattice_basis`（$B_{\mathbb Z}$，整数）、`cartesian_basis`（$Q$，正交）
  与 `coefficient_transform`（$R$，满足 $C = K_n B_{\mathbb Z} = QR$）取代原先含义含混的 `basis`。拟合参数是
  $Q$ 的系数，因此参数**取值**会变，而轨道数、不变量维数与交付的力常数不变。
- `pivots` 改为 `observation_rows`，并新增 `observation_matrix` 与 `observation_condition`。这些行是有限差分
  计划观测的分量，按观测块体积最大化选取；`reconstruct_sparse` 显式求解
  $Q_{\mathrm{obs}}\theta = y_{\mathrm{obs}}$，不再假设观测分量等于参数。
- realization identifiability 用模素数证书对精确整数 realization 矩阵求秩：不再有系数过滤、秩容差，
  也不再有无理晶胞的浮点回退。
- `TensorAction` 携带每次操作的 lattice 旋转，稳定子去重、复合与求逆都变成精确整数运算；`round(...,12)`
  的浮点签名删除。
- 轨道成员通过 `LatticeFrame.source_labels` 精确映射回用户晶胞（整数），reference 超胞、`sow`/`reap` 计划
  与 I/O 仍然指向同一物理 interaction。
- `ReferenceFrame` 携带该计算的 `lattice_frame`；`build_primitive_interaction_space` 用 `frame=` 取代
  `symmetry=` 与 `tolerance`。
- `cutoff=None`、group LASSO/ADMM 拟合以及其他无关能力均未改动。

### 修复

- 大 unimodular shear 不再让三、四阶整数代数膨胀：规范代数参考系使等价表示给出相同的整数基、观测行与条件数，
  并给出相同的有限差分重建结果。
- 文档：对称性与轨道理论页说明 lattice 参考系、两套轨道基与观测行；文档测试现在会运行
  `scripts/check_docs.py`，因此仓库的数学分隔符规则与中英镜像由测试套件强制。


## 4.0.0a5 — 2026-08-24

### 变化

- 有限差分工作流正式更名为 `FiniteDifferenceCalculation`，旧名称及兼容别名全部删除。
- 新增统一的公共 `perturb_structures()`，支持笛卡尔高斯采样和谐振模采样；SSCHA 复用
  同一个内部谐振采样器。
- 拟合、旋转修正和 SSCHA 的诊断量直接并入各自结果对象，不再维护 diagnostics 包装层。
- 软件统一使用 `mlfcs` 标准 logger；默认将 `INFO` 及以上消息写入 stdout，`DEBUG` 通过
  Python 标准 logging 接口开启。
- 顶层命名空间收缩为文档明确列出的公共工作流函数与类。

## 4.0.0a2 — 2026-08-14

### 新增

- 新增可合并写出 FC2--FC4 的 ALAMODE FCSXML，严格控制 MLFCS 原子顺序与原胞映射，
  并完整记录上游 ALM writer 的来源和许可证。

### 修复

- 高阶验证预测改为流式执行与 Gram 构造相同的有界物理设计分组，避免 JAX lowering
  将完整相互作用参数化捕获为数 GB 常量。

## 4.0.0a1 — 2026-08-14

### 新增

- 基于 compact FC2 的原生相容 q 点采样，同时支持量子和经典谐振系综；
- 显式虚频策略、频率过滤、采样诊断和可选的逐原子径向位移裁剪，默认不裁剪；
- 解析谐振模型测试以及仅开发环境运行的独立 phonopy 采样参考。
- 使用 phonopy 官方 pypolymlp 势函数和夹具的端到端 KCl SSCHA 参考。
- 新增中英文配对的算法说明，明确有限差分轨道完整性、位移键压缩契约及其数值稳定性
  边界。

### 变化

- `mlfcs.sscha` 改用共享的 MLFCS 对称约化 Gram 拟合器求解 FC2，并复用统一
  力常数 I/O；phonopy 和 symfc 不再是运行时依赖。
- canonical 迭代现在派生相互独立且可复现的子种子；笛卡尔初始化轮不再报告统计上
  未定义的 SSCHA 自由能。
- 轨道发现现在先规范化每个满足截断的候选，再对代表去重；当锚定候选集对规范化不
  闭合时，不再错误删除周期边界轨道。
- 标签对称的满秩主元会复用一个位移构型返回的全部受力响应。修复后的 K4As4Pt2 最大
  MIC FC3 方案由旧的 6636 个冗余构型缩减为 4244 个；错误的中间 4160 构型方案被明确
  废弃。

### 兼容性

- 不能将 4.0 以前生成的有限差分 `sow()` 方案与 4.0 的 `reap()` 混用。构型数量或顺序
  发生变化时必须重新生成完整的有序结构列表。

## 3.1.0 — 2026-08-09

### 新增

- 二阶力常数可通过 `rotational_sum_rule=True` 主动开启 Born–Huang 旋转求和规则；在
  联合阶数 API 能表达高阶耦合约束前，其他阶数会明确拒绝该选项；
- 平移与旋转约束使用一次联合稀疏 LSMR 投影；
- 求和规则投影前后默认报告 phonopy 风格的最大 drift；
- `cutoff=None` 表示当前超胞可枚举的最大相互作用半径；
- 新增中英文配对的求和规则文档。

### 变化

- 所有参数规模的平移 ASR 统一使用稀疏、矩阵无关的 LSMR 路径，删除稠密 Gram
  构造和规模切换阈值。

## 3.0.0 — 2026-08-03

3.0 是 MLFCS 的完整 ASE-first 重构：原有的分阶实现被统一的阶数参数化 API 和数值
流程取代。

### 新增

- `order >= 2` 的统一有限差分力常数流程；
- ASE Calculator 直接运行和确定性的外部 `sow()` / `reap()`；
- 递归中心有限差分模板和位移键去重；
- 对称性展开的稀疏力常数与惰性稠密物化；
- Gram 零空间和稀疏 LSMR 实现的严格平移 ASR；
- JAX 高阶张量操作的 CPU/GPU 选择；
- 任意阶通用稀疏 HDF5；
- phonopy FC2、phono3py FC3 HDF5 以及 ShengBTE FC3/FC4 输出；
- 使用显式周期几何的 ShengBTE FC3/FC4 导出；
- 可选的 phonopy/symfc 随机有效谐波模块；
- phonopy、phono3py、hiphive 转换、ShengBTE 与解析 Morse FC4 科学参考；
- Python 3.12/3.13 串行科学 CI。

### 变化

- 公共接口统一使用 ASE `Atoms` 和用户持有的 ASE calculator；
- 力生成不再绑定某个电子结构程序或机器学习势；
- 重建和默认保真导出共享同一个周期团簇几何；
- 3.0 只提供 Python API，不再提供 CLI。

### 兼容性

- 旧脚本需要迁移到 `FiniteDifferenceCalculation`、`sow()`、`reap()` 或 `run()`；
- `v3.0.0` 之前的标签属于旧实现或开发快照，仅为追溯保留，不属于 3.0 API 契约。
