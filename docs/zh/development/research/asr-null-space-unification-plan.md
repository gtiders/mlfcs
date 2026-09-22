---
title: ASR 统一后投影重构计划
audience:
  - advanced
  - developer
status: research
code_verified: 4.0.0a6
localized_only: true
---

# ASR 统一后投影重构计划

## 0. 决策、基线与交付方式

本计划取代旧的“ASR null space 进入拟合坐标系”方案。最终决策是：

1. 拟合与有限差分都先产生完整物理参数 $\theta_0$；
2. 两条路径随后调用 `constraints` 包拥有的同一个 ASR 正交投影器；
3. ASR 不参与 design 编译、Gram 累积、预条件或线性求解；
4. 同一份物理 Gram 必须能够分别产生“未投影”和“应用 ASR 投影”的结果；
5. 不保留旧 API、旧 Gram schema 或旧 null-space 兼容层。

实施时不得 rebase 或整体合并 `asr-null-space@134dee8`。它是研究原型，只能作为测试、文档和诊断字段的只读参考。正确起点是实施时最新的 `dev`：

```bash
cd /home/gwins/codespace/mlfcs-new
git fetch --all --prune
git worktree add .worktrees/asr-projection -b asr-projection dev
cd .worktrees/asr-projection
```

若不能联网，只省略 `git fetch`，但必须在交付报告中记录实际 `dev` commit。开始修改前记录：

```bash
git rev-parse dev
git status --short
pytest -q tests/test_reconstruction_asr.py tests/test_fitting_constraints.py
```

主仓库已有的未提交图片、研究文档与教程文件不得复制、暂存或清理。

## 1. 物理与数值契约

### 1.1 唯一公开语义：求解后投影

设 $X$ 是完整物理参数空间的力设计矩阵，$f$ 是训练力，$A_n$ 是第 $n$ 阶 IFC 的平移 ASR 方程。

拟合先求无约束解：

$$
\theta_0=\arg\min_\theta\lVert X\theta-f\rVert_2.
$$

随后每一阶独立做欧氏正交投影：

$$
\theta_n
=\arg\min_{\vartheta}\lVert\vartheta-\theta_{0,n}\rVert_2,
\qquad
A_n\vartheta=0.
$$

有限差分先由 observation rows 重建 $\theta_{0,n}$，随后执行完全相同的第二个问题。两条路径必须共享：

- 同一个 $A_n$ 构造函数；
- 同一个投影算法；
- 同一个相对停止准则；
- 同一组投影前后残差和修正量定义；
- 同一种失败策略。

### 1.2 明确不做严格受约束拟合

本计划不求解：

$$
\min_{A\theta=0}\lVert X\theta-f\rVert_2.
$$

因此不得把 $\theta=Nz$ 写入拟合 design，也不得使用 Gram 度量投影冒充本计划的欧氏投影。一般情况下，“无约束拟合后欧氏投影”与“受约束最小二乘”不是同一个估计量。文档、日志和测试必须如实使用“post-fit ASR projection”或“拟合后 ASR 投影”，不得称为 constrained fit。

选择该语义的理由是：ASR 在这里是对最终 IFC 的统一物理修正，而不是训练估计器的一部分；有限差分与拟合因此具备真正相同的处理顺序。教程证据显示两种路线的力误差只在较后有效数字变化，但新实现仍必须重新量化这一差异。

### 1.3 参数度量

$\theta$ 必须始终是 primitive orbit 的 `cartesian_basis` 系数，按当前物理参数顺序拼接。投影采用该坐标中的欧氏范数。实现前新增测试确认消费方使用的 `cartesian_basis` 列正交归一；若某个合法 orbit 不是正交归一基，不得静默继续，必须先把投影改写成对应的 Cartesian tensor 度量并记录设计变更。

ASR 按 IFC 阶数独立。FC2、FC3、FC4 之间不得出现联合 null space、跨阶约束或 block-diagonal lift。

## 2. 从 `asr-null-space@134dee8` 保留什么

这里的“保留”指在最新 `dev` 上重新实现或选择性移植思想，不是 cherry-pick 原提交。

| 原型成果 | 决定 | 新实现中的落点 |
|---|---|---|
| 跨表示 ASR oracle 与展开后 IFC 检查 | 保留并简化 | 移植到 `tests/test_asr_projection.py`，验证 orbit 参数残差和展开后 atomic sum 同时下降 |
| 投影幂等、最小修正、零空间输入不变测试 | 保留 | 针对共享投影器测试，不再构造 `TranslationalNullSpace` |
| `ProjectionResult` 的诊断思想 | 保留 | 改成小型 `ASRProjectionResult` |
| 投影前后最大漂移、相对修正量日志 | 保留 | fitting 与 finite difference 使用同一报告字段 |
| `maximum_asr_residual` 结果字段 | 保留并拆成 before/after | 避免只报告投影后接近零的数字 |
| Gram merge/save/load 的身份检查思想 | 保留但解除 ASR 绑定 | 只记录完整物理 design 身份，见 §6 |
| 删除 `ForceConstantFitter.evaluate_force_error` | 保留 | 单独提交，消除 `fitting -> calculators` DAG 违规 |
| 英文 `constraints-api.md` 修复 | 保留 | 从错误的 versioning 占位内容改成真实约束 API 页面 |
| 教程中的误差和耗时数据 | 仅作比较基线 | 不复制旧产物；最终从最新 `dev` 重跑 |

## 3. 必须删除或回退什么

以下原型内容全部不进入新分支：

1. `src/mlfcs/constraints/null_space.py`；
2. `TranslationalNullSpace`、`IdentityParameterSpace`、`JointParameterSpace` 和 `ParameterSpace`；
3. integer kernel、HNF、SNF、模素数 rank certificate、numeric gauge、free columns；
4. `reduce_design()`、`lift_parameters()`、`independent_parameters`；
5. `parameter_space_fingerprint`、`parameter_space_policy`；
6. Gram metadata 中的 `acoustic_sum_rule` 和 `independent_parameter_count`；
7. `prepare_gram(acoustic_sum_rule=...)`；
8. design operator 的 `parameter_map` 或 `parameter_space`；
9. 只在 independent coordinates 上工作的 Gram 与 solver；
10. 旧分支的 38 个教程产物和旧 `src/mlfcs/phonon/**` 修改；
11. 为兼容上述 schema、字段或导入路径而增加的任何 alias。

当前 `dev` 中已有的 `fitting/constraints.py`、`explicit_constraint_null_space`、`ConstraintNullSpace` 和 projected-CG 分支也应在新实现完成后删除。它们属于被替换的受约束拟合路线，不能与后投影路线并存。

## 4. 共享投影器设计

### 4.1 文件与最小 API

所有实现收敛在 `src/mlfcs/constraints/translational.py`，不新建 1000 行级别的参数空间模块。建议 API：

```python
@dataclass(frozen=True, slots=True)
class ASRProjectionResult:
    parameters: np.ndarray
    initial_residual: float
    final_residual: float
    correction_norm: float
    relative_correction: float
    optimality_residual: float
    iterations: int


@dataclass(frozen=True, slots=True)
class TranslationalASRProjector:
    order: int
    physical_dimension: int
    constraints: sparse.csr_matrix

    @classmethod
    def from_orbit_space(cls, orbit_space) -> "TranslationalASRProjector": ...

    def maximum_residual(self, parameters: np.ndarray) -> float: ...

    def project(
        self,
        parameters: np.ndarray,
        *,
        tolerance: float,
    ) -> ASRProjectionResult: ...
```

可以继续公开 `build_translational_constraints`，但 `project_parameters` 与 `project_acoustic_sum_rule` 的重复入口要么收敛到该对象，要么删除。生产代码只能有一个投影算法。

### 4.2 投影算法

令 $r=A\theta_0$。求满足 $A\delta=r$ 的最小范数修正 $\delta$，然后：

$$
\theta=\theta_0-\delta.
$$

可使用 SciPy `lsmr(A, r)`，因为它返回该一致系统的最小范数解。不得显式构造稠密 projector、$AA^T$ 的逆或 null-space lift。

要求：

- 空参数和零约束返回输入副本，迭代数为零；
- 非有限输入在进入 SciPy 前拒绝；
- shape 不符时明确报告期望维数；
- LSMR 未达到统一相对残差界时抛出异常，不返回“接近满足”的结果；
- 最多允许少量 iterative refinement，但停止准则只能有一套；
- `optimality_residual` 验证修正属于 $\operatorname{range}(A^T)$ 所需的一阶条件；
- 不把 solver `istop` 单独当成功，必须检查实际 ASR 残差。

### 4.3 只保留两个数值策略

本模块只允许两类数值策略，且必须有名字、文档和测试：

1. `COEFFICIENT_ZERO_RTOL`：清理 `cartesian_basis` 变换中机器舍入产生的代数零。它必须按当前矩阵尺度乘机器精度推导，例如 $c\epsilon\max(1,\lVert B\rVert_\infty)$，不得继续使用无尺度固定 `1e-12` 判定系数是否为零；
2. 调用方显式传入的 `tolerance`：投影相对停止精度。残差验收使用归一化形式，例如

$$
\lVert A\theta\rVert_\infty
\le
\text{tolerance}\,
\max(\lVert A\rVert_\infty\lVert\theta\rVert_\infty,\text{tiny}).
$$

`lsmr` 的 `atol`、`btol` 必须由同一个 `tolerance` 派生，不得形成第三套独立策略。该 tolerance 是线性代数相对精度，不是长度精度，因此不得复用 `symprec`。

## 5. 拟合路径改造

### 5.1 Gram 始终处于完整物理参数空间

`ForceDesignOperator` 必须删除：

- `parameter_map`；
- `parameter_space`；
- `reduce()` 中的 ASR 映射；
- `fit_n_parameters` 与物理参数数目不同的可能性。

每个 snapshot 的 design 直接进入 Gram：

$$
G=X^TX,
\qquad
b=X^Tf.
$$

`prepare_gram(structures)` 不接受 ASR 参数。一次生成的 Gram 可以多次用于：

```python
raw = fitter.fit(gram, acoustic_sum_rule=False)
physical = fitter.fit(gram, acoustic_sum_rule=True)
```

两次调用不得重新编译 design 或重新扫描训练结构。

### 5.2 `fit()` 的处理顺序

建议签名：

```python
def fit(
    self,
    gram: GramStatistics,
    *,
    acoustic_sum_rule: bool = True,
    asr_tolerance: float = 1e-10,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    precondition: bool = True,
    allow_unconverged: bool = False,
) -> FittingResult:
```

严格按以下顺序：

1. 校验 Gram 的物理 design identity；
2. 在完整参数空间求无约束解 `unprojected_parameters`；
3. 记录无约束 normal-equation residual 和训练误差；
4. 若启用 ASR，按 order slice 调用对应 `TranslationalASRProjector.project()`；
5. 拼接 `fitting_parameters`；
6. 用同一物理 Gram 对投影后参数重新计算 RMSE、相对误差和按阶 force RMS；
7. 只用投影后参数展开 IFC；
8. metadata 记录 ASR 是否应用、投影精度、投影前后残差和修正量。

投影后参数通常不再满足无约束 normal equation。结果对象必须避免歧义：

```python
class FittingResult:
    force_constants: ForceConstants
    fitting_parameters: np.ndarray
    unprojected_parameters: np.ndarray
    parameter_scale: np.ndarray
    gram_statistics: GramStatistics
    training_force_rmse: float
    training_relative_force_error: float
    unprojected_training_force_rmse: float
    unprojected_training_relative_force_error: float
    solver_normal_equation_residual: float
    maximum_asr_residual_before: float
    maximum_asr_residual_after: float
    asr_parameter_correction: float
```

可以调整具体字段名，但不得把无约束 solver residual 描述成投影后参数的 stationarity residual，也不得只保留投影后的训练误差而隐藏投影代价。

### 5.3 求解器清理

`fitting/linear_solvers.py` 只保留无约束 Gram 求解。删除：

- `explicit_constraint_null_space`；
- `ConstraintNullSpace`；
- constraint matrix 参数；
- projected CG；
- 求解结束后的 constraint patch。

预条件继续在物理列上计算。若某列无观测信息，沿用明确的 inactive-column 策略并覆盖零维/全零 Gram 测试。

## 6. Gram 身份：保留思想，但不绑定 ASR

旧原型正确指出：仅凭 shape 不能安全 merge 两份 Gram。新 schema 只描述完整物理 design，至少包含：

- `design_schema`；
- `design_fingerprint`；
- `physical_parameter_count`；
- `orders`；
- primitive/reference/frame 身份；
- cutoff、max body order；
- orbit 顺序和每个 orbit 的物理参数维数；
- 编译 design plan 的稳定身份。

不得包含：

- `acoustic_sum_rule`；
- projection tolerance；
- independent parameter count；
- null-space/gauge/free-column 身份；
- `parameter_map` 数组。

`merge`、`save`、`load` 和 `fit` 都校验 physical design identity。旧 Gram 文件直接按开发期破坏性变更拒绝，不写兼容读取器。

若 design fingerprint 无法在不引入新的大规模规范化系统的前提下完成，应把它拆成独立后续提交；不得因此恢复 null-space fingerprint，也不得阻塞核心后投影改造。

## 7. 有限差分路径改造

有限差分保留“先重建、后投影”的现有顺序，但不再接受通用 `parameter_space` 对象。

建议：

```python
reconstruct_sparse(
    ...,
    acoustic_sum_rule: bool = True,
    asr_tolerance: float = 1e-10,
)
```

或者由 `FiniteDifferenceCalculation` 缓存每阶 `TranslationalASRProjector` 并显式传入 projector。无论采用哪一种：

- projector 只能来自 `constraints.translational`；
- 禁用 ASR 时直接保留原始重建参数，但仍可报告初始漂移；
- 启用时记录与拟合完全同名、同单位的 before/after/correction 字段；
- 不记录 parameter-space fingerprint；
- displacement manifest 身份不应包含 ASR，因为 ASR 不改变 displacement plan；
- ASR 策略只影响 `reap/reconstruct` 的结果 metadata。

## 8. DAG 与 reciprocal/SSCHA 调用方

### 8.1 DAG 修复

删除 `ForceConstantFitter.evaluate_force_error`，因为它延迟导入 `mlfcs.calculators`，形成 `fitting -> calculators`。若教程需要外部数据误差，应在教程中用公开 calculator 或直接用 Gram 指标；不在 `fitting` 中建立替代导入。

架构测试必须恢复全绿，不能再把该失败列为基线例外。

### 8.2 SSCHA

当前生产路径已经从 `mlfcs.phonon` 迁到 `mlfcs.reciprocal`。只修改最新路径：

```text
src/mlfcs/reciprocal/sscha/solver.py
```

SSCHA 不再把 ASR 策略传给 `prepare_gram`，而是在 `fit` 时传：

```python
gram = fitter.prepare_gram(structures)
result = fitter.fit(gram, acoustic_sum_rule=True)
```

不得恢复 `src/mlfcs/phonon/**`，不得让 reciprocal 依赖 `mlfcs.tools`。

## 9. 测试先行

### P0：在修改生产代码前建立红色契约测试

新增或重写测试，至少覆盖：

1. 同一份 Gram 可先后执行 ASR 关闭和开启；
2. `prepare_gram` 签名中不存在 `acoustic_sum_rule`；
3. design/Gram 维数始终等于物理参数数目；
4. fitting 与 finite difference 对同一人为参数向量调用同一个 projector，结果逐位相同；
5. 投影幂等：$P(P\theta)=P\theta$；
6. 已满足 ASR 的输入不变；
7. 投影修正是满足 ASR 的最小欧氏修正，并与小矩阵 dense oracle 对拍；
8. 零约束、空参数、全零参数和冗余约束有定义；
9. NaN/Inf 在 SciPy 前拒绝；
10. 不收敛时失败，消息包含 order、维数、初始和最终归一化残差；
11. 投影后的展开 IFC 满足直接 atomic-sum oracle；
12. Gram merge 拒绝相同 shape、不同 physical design 的统计量；
13. `fitting` import closure 不到达 `calculators`；
14. `src/mlfcs/phonon` 与 `src/mlfcs/constraints/null_space.py` 不存在。

必须保留一个测试明确证明：后投影与严格受约束拟合在一般矩阵上结果不同，防止未来维护者又把两种语义混为一谈。

## 10. 性能与资源验收

后投影会使 Gram 回到完整物理维数。这是简化架构的真实成本，不能用旧分支的 reduced-Gram 数据代表新实现。

至少测量：

| 案例 | 物理参数数 | 旧约化参数数（仅参考） |
|---|---:|---:|
| K4As4Pt2 | 6849 | 4784 |
| SnSe | 8572 | 5801 |
| Ba8Ga16Ge30 | 8329 | 3412 |

对每个案例报告：

- Gram shape 与理论内存 $8p^2$；
- 峰值 RSS；
- design、BLAS、solve、ASR projection 分项时间；
- 投影迭代数；
- 投影前后 RMSE/相对误差；
- ASR 残差与参数相对修正。

验收原则：

1. 不得 OOM；建议在交付环境中峰值 RSS 不超过 1.5 GiB；
2. projector 只能在求解后每阶运行一次，不得进入 snapshot 循环；
3. 不得为了恢复 reduced-Gram 性能重新引入 null-space design；
4. 若完整 Gram 成为不可接受的资源瓶颈，应暂停并单独设计 matrix-free 无约束拟合，不得在本计划中偷偷回退旧架构；
5. 性能退化和简化收益都要量化，不以“测试通过”代替成本报告。

## 11. 教程与外部依赖

最终实现稳定后，从最新 `dev` 重跑所有受到结果字段或拟合流程影响的独立任务。每个任务脚本必须覆盖写入本目录固定 `fit.log`，完整保存 stdout、stderr 与 traceback，并提交日志。

不得 cherry-pick `db7249a` 的旧产物。SCPH/SSCHA 路径已经变化，必须重新运行当前脚本。

`tutorial/Si/force-fitting-ase` 的 NEP3 模型与可安装 calorine $\ge 1.0$ 不兼容，是明确阻塞项：

1. 先核对仓库声明的 calorine 版本与模型来源；
2. 不得静默把 NEP3 换成另一物理模型；
3. 若没有可复现的兼容环境，必须单独报告并由维护者决定“固定旧 calorine”还是“更新模型与基准”；
4. 在作出决定前不得声称全部教程已重跑，也不得用旧 `fit.log` 冒充新结果。

教程验收不仅比较最终相对误差，还要记录 ASR 投影带来的误差增量：

$$
\Delta\mathrm{RMSE}
=\mathrm{RMSE}(P\theta_0)-\mathrm{RMSE}(\theta_0).
$$

## 12. 文档与 API

更新中英文：

- `theory/constraints.md`；
- `reference/constraints-api.md`；
- `reference/fitting-api.md`；
- `reference/finite-difference-api.md`；
- `reference/sscha-api.md`；
- 双 CHANGELOG。

文档必须明确：

1. ASR 是求解后欧氏投影；
2. 它不改变 Gram/design；
3. 同一个 Gram 可以比较投影开/关；
4. 投影会轻微改变训练误差；
5. solver normal residual 属于未投影解；
6. `asr_tolerance` 是线性代数相对精度，不是 `symprec`；
7. 不再存在 public null-space/gauge/independent-coordinate API。

所有 Markdown 数学公式只使用 `$...$` 与 `$$...$$`。

## 13. 提交拆分

从最新 `dev` 形成以下可独立审查的提交：

### A. `test: characterize post-fit ASR projection`

- 只新增契约测试和小型 dense oracle；
- 在旧行为上应有明确红灯；
- 不复制 440 行以上的旧 null-space 实现测试。

### B. `constraints: make ASR one shared post-processing projection`

- 实现 `TranslationalASRProjector` 与 `ASRProjectionResult`；
- 收敛重复投影入口；
- 用尺度化机器精度清理代数零；
- finite difference 迁移到共享入口。

### C. `fit: solve the physical Gram before applying ASR`

- design operator 删除 parameter map；
- `prepare_gram` 始终生成物理 Gram；
- solver 删除约束/null-space 分支；
- `fit` 后投影并重算指标；
- 同一 Gram 支持 ASR 开/关。

### D. `fit: bind Gram files to the physical design`

- 只加入 physical design identity；
- merge/save/load/fit 校验；
- 不含 ASR 身份。若该项需要额外大型规范化，应独立延期，不阻塞 B/C。

### E. `architecture: remove the fitting-to-calculators edge`

- 删除 `evaluate_force_error`；
- 更新调用者与 DAG 测试；
- 不夹带 ASR 算法改动。

### F. `docs: document ASR as a shared post-solve projection`

- 中英文理论/API/CHANGELOG；
- 修复英文 constraints API 占位页。

### G. `tutorials: regenerate results after ASR projection`

- 最后提交所有成功重跑的脚本和产物；
- NEP3 阻塞若未解决则不得创建虚假的完整产物提交。

任何提交都必须在自己的范围内通过聚焦测试；不得把 A–G 压成一个大提交。

## 14. 最终验证

```bash
pytest -q tests
ruff check src tests scripts
ruff format --check <本分支修改的 Python 文件>
python scripts/check_docs.py
python scripts/sync_readmes.py --check
mkdocs build --strict
```

源码门禁：

```bash
test ! -e src/mlfcs/constraints/null_space.py
test ! -e src/mlfcs/phonon
rg -n "TranslationalNullSpace|IdentityParameterSpace|JointParameterSpace|parameter_space_fingerprint|parameter_map|explicit_constraint_null_space|ConstraintNullSpace" src tests
rg -n "prepare_gram" src tests tutorial | rg "acoustic_sum_rule|parameter_space"
```

第一条 `rg` 除专门验证旧 API 已删除的负面测试外必须无匹配。最终报告必须包含：

1. 实际基线与 HEAD；
2. A–G 提交列表；
3. 删除的 API/schema；
4. 投影数学语义；
5. 两个数值策略及其来源；
6. 同一 Gram 切换 ASR 的测试证据；
7. fitting/finite difference 同投影器证据；
8. 三个真实案例的内存、时间和误差变化；
9. 所有教程重跑清单及 `fit.log`；
10. 未完成项和外部依赖阻塞，不得用“基线既有”隐藏。

## 15. 停止条件

出现以下任一情况应停止实现并报告，不得自行恢复 null-space 拟合：

1. 合法 orbit 的 `cartesian_basis` 不是正交归一，导致欧氏系数投影没有既定物理度量；
2. 完整物理 Gram 在目标教学案例上超过资源上限；
3. ASR 投影无法在统一相对容差内收敛；
4. 投影后训练误差出现数量级恶化，而不是末位变化；
5. 展开后 IFC 的直接 atomic-sum oracle 与参数空间残差矛盾；
6. NEP3 教程无法建立可复现环境但交付要求全部教程重跑；
7. 实现需要重新引入 gauge、free coordinates 或 ASR-specific Gram 才能工作。

这些情况意味着需要维护者重新选择物理度量、资源策略或教程模型，不属于实现者可以静默改变的细节。
