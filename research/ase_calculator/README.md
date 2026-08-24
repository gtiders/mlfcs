# 从 HDF5 到 ASE Calculator 的源码级调研

## 1. 结论

结论分成两个层次：

- 对“把当前 HDF5 解释成 $E_0=0$、FC1=0 的 canonical Taylor 多项式”而言，属于
  **Small Refactor**。HDF5 中的 FC2 及以上已经足够，核心 evaluator 不需要 fitting、训练数据、
  covariance、symmetry、JAX 或 design matrix。
- 对“reload 后严格重现当前 Wick validation predictor”而言，属于
  **Schema/Architecture Change Required**。当前 HDF5 没有保存 Wick→Taylor 产生的 FC1，且
  FC2+FC3+FC4 的 intertwiner 原型还发现了一个独立的约 $3.9\times10^{-3}$ force 差异，必须
  先审计后才能把 validation 统一到新 evaluator。

因此本轮不应直接发布公共 `MLFCSCalculator`。独立原型证明 evaluator 与 ASE adapter 本身可行，
也证明 FC1 可以作为 primitive-site source-independent Taylor 项保存并展开到任意合法超胞。

## 2. 当前 validation 的真实调用链

当前 fitting validation 不经过 `ForceConstants`：

```text
训练结构
  → StructureRelation.displacement()
  → FitDataset.displacements
  → symmetrized_covariance()
  → OrderParameterization
  → ForceDesignOperator / JAX Wick feature kernel
  → X theta_Wick
  → RMSE
```

关键实现为：

- `structure/relation.py::StructureRelation.displacement()`：验证相同原子数、原子顺序和晶格，
  再计算 MIC displacement；
- `fitting/dataset.py::FitDataset.from_atoms()`：从训练 force 减去 reference force；
- `fitting/design.py::ForceDesignOperator.matvec()`：直接预测 validation force；
- `fitting/design.py::predict_force()`：低层 Wick force contraction；
- `fitting/fitter.py::ForceConstantFitter.fit()`：训练 metrics 从 streamed Gram 得到，validation
  metrics 使用 `validation_operator.matvec(parameters_numpy)`。

拟合完成后才执行：

```text
theta_Wick
  → build_wick_to_taylor_transform()
  → theta_Taylor
  → expand_fitted_orders()
  → SparseOrderForceConstants
  → ForceConstants
  → HDF5 v3
```

所以当前同时存在两种 force 语义：训练期 prediction 使用 Wick coordinates；保存结果使用
canonical Taylor IFC。当前没有从 sparse IFC 反向服务 validation 的正式 evaluator。

## 3. 当前势能的数学定义

对保存的最高阶 $p$，canonical sparse IFC 自然定义 reference-relative 多项式

$$
\Delta E(u)=\sum_{n=1}^{p}\frac{1}{n!}
\sum_{a_0,R_0}\sum_{a_1,R_1\ldots a_{n-1},R_{n-1}}
\Phi^{(n)}_{a_0\ldots a_{n-1}}(R_1-R_0,\ldots,R_{n-1}-R_0)
\prod_{k=0}^{n-1}u_{a_k,R_k}.
$$

第一原子使用零平移锚点，有限 reference evaluation 时遍历所有 anchor cell。force 为

$$
F_{a\alpha}(u)=-\frac{\partial\Delta E}{\partial u_{a\alpha}}.
$$

源码中的 `force_design_batch()` 和 `predict_force()` 对每个 interaction 的每个 tensor axis
求 leave-one-axis derivative，并除以 $n!$；对全部轴求和后等价于常见的 $1/(n-1)!$ force
convention。原型对 sparse Taylor evaluator 与同一正式 design kernel 的 covariance-zero Taylor
evaluation 比较，最大 force 差为 $9.37\times10^{-16}$ eV/Å。

当前保存对象不包含：

- 常数能量 $E_0$；
- reference 的绝对势能；
- reference force；
- FC1。

所以现有 HDF5 的确定语义只能是 $E_0=0$、FC1=0 的 reference-relative Taylor polynomial。
它不能给出绝对 potential energy。stress 也没有严格定义，因为 cell strain 不属于当前固定晶格
位移展开，且没有 strain derivative IFC。

## 4. Wick 与 Taylor

Wick 是 fitting coordinates，不是 HDF5 的最终物理表示。`ForceConstantFitter.fit()` 明确写入：

```text
fitting_basis = wick
force_constants_basis = taylor
```

covariance、训练数据、Gram 和 solver 状态均不写入 HDF5，这一设计方向是正确的。但存在两个
必须先解决的细节。

### 4.1 被省略的 FC1

`basis/wick_taylor.py::omitted_taylor_fc1()` 返回

$$
\Phi^{(1)}_{a\alpha}=\left.\frac{\partial\Delta E}{\partial u_{a\alpha}}\right|_{u=0},
$$

其 force contribution 为 $-\Phi^{(1)}_{a\alpha}$。它是 `(n_primitive, 3)` 的笛卡尔数组，
没有 translation label。非中心对称 GaAs 原型得到的最大 FC1 为
$6.67\times10^{-3}$ eV/Å；Wick force 与不含 FC1 的 Taylor force 之差等于 $-\Phi^{(1)}$，
最大误差 $4.34\times10^{-18}$ eV/Å。补入 FC1 后，两条路径同样在
$4.34\times10^{-18}$ eV/Å 内一致。

FC1 能与当前任意 target supercell 逻辑兼容：primitive site 重排时同步重排；primitive lattice
幺模换基不改变笛卡尔向量；target supercell 中按 `primitive_index` 在每个 cell 重复。它不需要
Wigner、residue lift 或 cutoff。

但当前 `SparseOrderForceConstants` 不能直接表示 order 1：translations 的 reshape 使用
`(-1, order - 1, 3)`，对零长度轴无法推断行数。因此正式 schema 应使用独立、明确的
`reference_terms/fc1`，而不是把 FC1 偷塞进现有 sparse order group。研究性 HDF5 extension 在
primitive 和更大 target supercell 上 reload 误差均为零。

### 4.2 FC4→FC2 intertwiner 差异

在中心对称 Ar FC2+FC3+FC4 原型中 FC1 严格为零，但：

- Taylor design 与 sparse evaluator 相差 $1.17\times10^{-14}$ relative；
- Wick prediction 与当前 intertwiner 生成的 Taylor prediction 相差
  $3.90\times10^{-3}$ relative。

因此这不是 evaluator、factorial 或 FC1 导致，而是当前多阶 intertwiner/finite realization
还需要单独审计。正式 Calculator 不应掩盖或平均这项差异。

## 5. HDF5 v3 自包含性

当前 `io/hdf5.py` 保存：

| 字段 | 类别 | Calculator 用途 |
|---|---|---|
| `structures/primitive/{cell,positions,numbers,pbc}` | A | 定义 primitive reference |
| `force_constants/<order>/sites` | A | primitive site identity |
| `force_constants/<order>/translations` | A | exact primitive $R$ |
| `force_constants/<order>/tensors` | A | Cartesian Taylor tensor |
| `schema_version`, `units`, `tensor_basis` | A | convention 验证 |
| `force_constants_basis=taylor` | A/校验 | 拒绝非 Taylor artifact |
| cutoff、ASR、method、solver | C | provenance 与诊断 |
| fitting basis、regularization、training count | C | provenance |
| covariance、parameters、Gram、orbit basis | B | 仅 fitting；当前不保存且 evaluator 不需要 |

其中 A 表示 evaluation 必需，B 表示 fitting-only，C 表示 provenance。缺少的 evaluation 状态为：

- 若要求严格复现 Wick validation：FC1；
- 若要求恢复原始未减 reference force：reference force，但它是 source-supercell 特有量，不应
  混入力常数 polynomial；
- 若要求绝对能量：$E_0$；
- 若未来包含 observable closure：source-owned `FiniteHarmonicResponse` companion artifact。

HDF5 v3 没有 source reference，这是 canonical exact-$R$ schema 的有意设计。因此只传路径时
Calculator 最多默认在 primitive reference 上工作；要在原训练超胞或其他超胞工作，必须显式
提供目标 reference。

## 6. 最短 HDF5 → force 路径

原型路径为：

```text
read_hdf5()
  → realize_force_constants(fc, explicit_reference)
  → cache concrete atom tuples and tensors
  → StructureRelation.displacement(atoms)
  → NumPy polynomial contraction
  → Delta E, forces
  → thin ASE Calculator adapter
```

它不读取训练数据，不构造 fitter、orbit、design、Gram 或 solver，也不运行 spglib 于每个 MD
step。结构 relation 和 concrete realization 只在 Calculator 构造时建立一次。

现有 Si FC2、Si FC2+FC3、Si FC2+FC3+FC4 以及有限差分 Taylor HDF5 均可直接载入并产生有限
$\Delta E$ 和 force。ASE adapter 与 core evaluator 的差异约为 $10^{-15}$ eV/Å；数值 energy
gradient 与 analytic force 也通过中心差分检查。详细结果见 `results.json`。

## 7. 结构兼容性

建议正式 API 为：

```python
potential = ForceConstantPotential(force_constants, reference=reference)
calc = MLFCSCalculator(potential)
```

以及：

```python
calc = MLFCSCalculator.from_hdf5("mlfcs.h5", reference=reference)
```

`reference=None` 时只使用 HDF5 中的 primitive。规则应为：

- 同一 calculate stream 必须具有相同 atom count、species order 和 cell；
- 用户可在构造 Calculator 前显式重排 reference；Calculator 不静默重排每一步 Atoms；
- wrapped/unwrapped coordinate 由统一 MIC displacement 处理；
- lattice strain、不同 primitive、任意旋转和不同 cell 拒绝；
- 任意合法整数 target supercell 可以在构造时通过 exact-$R$ realization 支持；
- 整体平移不被偷偷减掉，其不变性应由 ASR 保证；
- displacement 超出 Taylor 有效域时可警告，但不能靠最近 reference site 静默重新标号。

## 8. 性能与 backend

正式 core evaluator 应在构造时预计算每个 order 的 concrete atom tuples、tensor 和 multiplicity，
每步只做 displacement 与多项式 contraction。默认 NumPy 最合适：

- 没有 JAX 首次编译延迟；
- `import mlfcs` 不会加载 JAX；
- 小规模 validation、debugging 和 phonon test 足够；
- 算法直接对应 sparse IFC 数学语义。

若以后面向长 MD，可在完全相同的缓存数据上增加可选 JAX backend，但不应让 ASE adapter 拥有
第二套物理实现。

## 9. 与未来 FiniteHarmonicResponse 的兼容

应保持一个 evaluator、两种 capability：

$$
\Delta E=\Delta E_{\mathrm{transferable}}^{(2+)}
+\Delta E_{\mathrm{FC1}}
+\Delta E_{\mathrm{finite\ harmonic}}^{\mathrm{source}}.
$$

transferable mode 可 realization 到任意合法 target；source-enhanced mode 只在 fingerprint 和
translation sublattice 匹配时叠加 finite harmonic residual。FC3/FC4 evaluator 不应复制。

## 10. 最小正式改动计划

在修复/解释 intertwiner 差异后，建议独立小提交：

1. `force_constants/potential.py`：纯 NumPy `ForceConstantPotential`，缓存 realization，返回
   reference-relative energy、forces 和 order-resolved contribution；
2. `calculators/ase.py`：薄 `MLFCSCalculator` adapter，仅支持 energy/forces；
3. HDF5 schema 增加可选 `reference_terms/fc1`，含 unit、Cartesian basis 和 Taylor-gradient
   semantics；不保存 $E_0$，除非以后存在可靠来源；
4. `FittingResult` 写出当前 `omitted_taylor_fc1`，并把命名改成明确的 Taylor FC1；
5. validation 同时计算 Wick 与完整 Taylor evaluator，先作为一致性 guard；通过后再决定是否
   删除重复 prediction 路径；
6. 测试 FC2、FC2+FC3、FC2+FC3+FC4、有限差分 Taylor、Wick、原子重排、MIC wrapping、目标
   超胞、cell strain 拒绝和 energy-force finite difference。

建议模块边界：core potential 不依赖 ASE fitting/symmetry；ASE adapter 依赖 core potential；HDF5
loader 返回 physical artifact。stress 暂不实现。

## 11. 对核心问题的直接回答

1. validation 当前基于 Wick physical parameters 和 JAX design operator，不基于最终 HDF5 IFC。
2. HDF5 保存 physical canonical Taylor IFC，不保存 fitting coordinates。
3. Wick HDF5 对 FC2 及以上脱离 covariance；但未保存 FC1，不能严格序列化完整 predictor。
4. 当前 HDF5 能唯一规定零 $E_0$、零 FC1 的 Taylor polynomial。
5. 可以直接 HDF5→evaluator→ASE Calculator，但原训练超胞需显式 reference。
6. Calculator 不需要 fitting/design machinery。
7. 每步 evaluation 不需要 symmetry；构造 target relation 时需要一次结构验证。
8. 支持 primitive 或显式验证的整数 target supercell；不支持应变、旋转和任意 cell。
9. energy 是 reference-relative $\Delta E$，不是 absolute energy。
10. stress 当前没有严格定义，不应实现。
11. 最合理抽象是与 ASE 无关的 `ForceConstantPotential`。
12. 最小正式文件是 core potential、ASE adapter、HDF5 FC1 extension 及测试。
13. 最终可以统一 validation，但必须先解决 FC1 序列化和 intertwiner 差异；现在直接替换不安全。

本轮结论为：**Small Refactor for the zero-FC1 Taylor artifact；Schema/Architecture Change
Required for exact fitted-predictor serialization**。不是 No-Go。
