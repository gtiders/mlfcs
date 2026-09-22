# 整数晶格轨道重构：合并前收尾要求

## 1. 决策前提

本文接续 `integer-lattice-orbit-refactor-plan.md`，用于把 `reflector` 分支继续修改到可以合并。

本轮明确采用以下产品与架构决策：

1. 项目仍处于开发阶段，允许破坏 Python API、参数文件和有限差分计划格式。
2. 不提供旧字段、旧构造签名、旧参数语义或旧 `sow()` 计划的兼容层。
3. 禁止“看似兼容但语义已经改变”的别名，例如不能让旧 `basis` 静默返回新 `cartesian_basis`。
4. 最重要的契约是规范性、物理不变性、显式失败和可重复验证。
5. 等价晶格表示必须产生同一个精确代数对象，并产生相同的物理 Cartesian 子空间与物理预测。
6. 旧格式如果无法被正确解释，必须可靠拒绝，不能根据相同数组形状或配置 ID 猜测含义。

因此，当前 `PrimitiveInteractionOrbit.basis/pivots`、旧 `ReferenceFrame` 构造签名以及旧拟合参数坐标均不需要恢复。中英文 CHANGELOG 必须将它们登记为有意的破坏性变更。

## 2. 当前实现中已经成立的部分

以下内容已经过独立审阅，可以保留：

- primitive lattice 先进入规范 Minkowski 规约参考系；
- source cell 与 algebra cell 之间保存显式 unimodular 整数映射；
- 稳定子约束直接堆叠，不构造 $M^\mathsf{T}M$；
- 模素数秩证书支持 Python arbitrary-size integer 输入；
- 整数核由 Smith 标准形构造，并进行精确核验证；
- 精确晶格基 $B_{\mathbb Z}$ 与 Cartesian 数值基 $Q$ 分离；
- 物理子空间通过

$$
C=K_nB_{\mathbb Z}=QR
$$

  渲染，其中 $K_n=(A^\mathsf{T})^{\otimes n}$；
- 拟合、有限差分、ASR 和力常数展开统一消费 `cartesian_basis`；
- realization identifiability 在整数参考系中精确判秩；
- large unimodular shear 不再导致三阶、四阶整数膨胀；
- 当前材料测试中的最大 `observation_condition` 约为 $4.92$。

不要推翻这些实现重新回到 Cartesian 浮点秩判断或 pivot-normalized 参数基。

## 3. 必须完成的阻断项

### 3.1 删除 `cutoff=None`

`cutoff=None` 根据有限 reference supercell 隐式选择 primitive interaction radius。这会使“primitive 相互作用模型是什么”依赖用于观测它的有限 reference，而不是由 primitive 模型输入独立决定。

新的规范要求是：

- 正浮点数明确表示以 Å 为单位的 cutoff；
- 负整数明确表示 primitive 邻居壳层；
- `None` 不再是合法值；
- 有限 reference 是否能辨识该模型由 realization identifiability 独立检查；
- reference 太小时抛出 `InteractionAliasingError`，而不是静默缩短模型。

实施范围：

1. 从 `FiniteDifferenceCalculation`、`ForceConstantFitter`、SSCHA 及内部 settings 的类型和文档中删除 `None`。
2. 删除 `resolve_primitive_cutoff(..., reference=...)` 中 reference-resolved 分支。
3. 构造阶段对 `None` 抛出清楚的 `TypeError` 或 `ValueError`，错误信息说明必须提供正半径或负壳层。
4. 更新所有教程，将 `None` 替换为当前案例实际解析出的显式 cutoff，并在 README 中解释其来源。
5. 删除只验证 `cutoff=None` 的测试，新增“不同 reference 不改变 primitive interaction space”的属性测试。
6. 不提供弃用周期、兼容别名或环境开关。

验收条件：同一 primitive、同一显式 cutoff 在不同可辨识 reference 上必须得到逐轨道相同的 primitive interaction space；较小且不可辨识的 reference 必须显式失败。

### 3.2 删除当前 scaled orbit-group LASSO/ADMM

旧 `scaled_group_lasso` 在列预条件后的参数坐标中施加组范数。轨道基从 pivot-normalized 基改成正交 $Q$ 后，该惩罚对应的优化问题已经改变；一般可逆换基下，组 LASSO 不保持物理解不变。

当前项目尚未给出一个明确的、与参数坐标无关的正则化物理泛函，因此不要仅为保留旧入口而继续暴露它。

本轮要求：

1. 从 `ForceConstantFitter.fit()` 删除 `regularization` 参数。
2. 删除 `solve_scaled_group_lasso`、ADMM 状态和仅由该路径使用的辅助代码。
3. 从 `FittingResult`、metadata、日志和文档中删除对应字段。
4. 删除旧 LASSO 行为测试，保留无正则约束最小二乘的完整测试。
5. CHANGELOG 明确说明该实验性算法因参数化依赖而删除，不描述为性能清理。
6. 不接受兼容字符串、警告后回退最小二乘或静默忽略参数。

将来若重新引入轨道稀疏正则，应先定义物理不变量。例如在正交基 $Q$ 下，代表张量满足

$$
\lVert Q\theta\rVert_F=\lVert\theta\rVert_2.
$$

新的惩罚必须在物理参数 $	heta$ 上定义；数值预条件只能改变求解坐标，不能改变被惩罚的泛函。

### 3.3 为有限差分计划建立强身份并拒绝旧计划

当前 `dev` 和 `reflector` 对同一个简单立方 FC2 生成相同数量的四个构型，但第二个 displacement key 分别是 $z$ 和 $y$。只检查配置数量和从零开始的 ID 会把旧力静默解释成另一方向。

因为本轮明确不兼容旧计划，正确处理不是恢复旧 observation rows，而是让新协议可靠拒绝没有新身份信息的结果。

#### 新计划身份

为每个有限差分计划定义不可变 manifest，至少包含：

- `schema_version`；
- primitive 和 reference 的结构指纹；
- order、显式 cutoff、body order、symprec 和 displacement；
- derivative backend 与 stencil；
- 规范 orbit 标识；
- 每个 orbit 的 `observation_rows`；
- 排序后的 displacement keys；
- 配置 ID、位移原子、方向、符号和步长；
- 对上述规范序列计算的稳定哈希。

禁止对 `repr()`、Python 对象哈希、内存布局或浮点原始字节直接哈希。浮点字段必须先按文档规定规范编码；整数和枚举使用固定顺序和固定宽度的文本或二进制表示。

#### `sow()` / `reap()` 新契约

推荐引入明确的数据对象，例如：

- `DisplacementBatch`：manifest 加带身份的 displaced structures；
- `ForceBatch`：plan fingerprint、配置 ID 与 forces。

要求：

1. `sow()` 返回或同时写出 manifest，所有结构携带相同 plan fingerprint 和各自 configuration ID。
2. `reap()` 只接受带 fingerprint 的新结果对象；裸 `ndarray`、裸 list 和仅有数字 ID 的 mapping 必须拒绝。
3. fingerprint、schema、ID 集合、原子数或 force shape 任一不符都在求导前失败。
4. `run(calculator)` 可以在内部构造 `ForceBatch`，不降低直接 ASE calculator 工作流的便利性。
5. 外部 VASP、Quantum ESPRESSO 等工作流必须在教程中演示保存 manifest、收集 forces 和构造 `ForceBatch`。
6. 旧 `sow()` 输出没有 fingerprint，必须明确报“旧计划不受支持，请重新生成全部位移构型”。

不要提供“如果没有 fingerprint 就按位置顺序继续”的回退；这正是需要消除的静默错误。

必须增加以下回归测试：

- 同一个 batch 往返成功；
- forces 打乱后按 ID 恢复成功；
- 旧裸数组被拒绝；
- fingerprint 缺失或不同被拒绝；
- 配置数相同但 observation rows 不同被拒绝；
- primitive/reference/cutoff/stencil 任一改变均被拒绝；
- manifest 序列化再读取后 fingerprint 不变。

### 3.4 明确定义哪些对象必须规范

“结果可重复”需要区分精确规范对象与数值坐标。

#### 必须规范且可精确比较

- `LatticeFrame.algebra_cell` 的选择规则；
- source/algebra unimodular 映射；
- canonical atom order 和 motif mapping；
- primitive orbit representative 与 image 顺序；
- 稳定子整数约束；
- 饱和整数核所表示的整数格；
- observation row 顺序；
- finite-difference manifest 和 fingerprint。

#### 必须物理等价，但不要求跨 BLAS 逐位相同

- 正交 Cartesian 基 $Q$ 的具体列；
- 浮点 QR/SVD 中间量；
- 拟合参数坐标；
- 浮点展开后的张量尾数。

这些对象应通过子空间投影和物理预测比较，而不是强制逐列比较：

$$
Q_1Q_1^\mathsf{T}=Q_2Q_2^\mathsf{T}.
$$

#### 规范整数核

Smith 标准形给出的核格是饱和的，但 transformation basis 不自动构成跨实现版本的规范列基。若 `exact_lattice_basis` 被承诺逐位稳定，应在 SNF 后增加有明确定义的 column-HNF 或等价规范形，并规定：

- pivot 位置；
- pivot 符号；
- 列排序；
- 零维和满维情形；
- arbitrary-size integer 到 `int64` 的明确边界。

新增属性测试：给同一个饱和核基右乘随机 unimodular 矩阵，规范化后必须逐位相同。不要只测试“相同约束恰好让同一 SymPy 版本返回相同 transformation”。

如果不打算承诺 `exact_lattice_basis` 的逐位规范性，则改为只公开规范的核格指纹，并把测试从 `array_equal` 改为双向整数格等价证明；两者必须选择一个并写入文档，不能依赖偶然行为。

### 3.5 observation rows 必须只依赖物理子空间

保留 max-volume 贪心选择，但补充以下定义与测试：

1. 输入必须是正交基；
2. 行枚举顺序固定为 Cartesian tensor component 的字典序；
3. 体积相同或在明确阈值内时选择最小行索引；
4. 最终行按升序保存；
5. 保存并检查 `observation_condition` 上限；
6. 对任意正交矩阵 $O$，`select_observation_rows(Q)` 与 `select_observation_rows(QO)` 必须相同；
7. 对等价 unimodular primitive 表示必须相同。

文档应称其为“greedy max-volume selection”，不能声称求得全局 max-volume 解。

### 3.6 修正 source/algebra 标签文档

`LatticeFrame.source_positions` 按 canonical atom order 保存，而 `source_labels()` 返回的 site 是 source atom index。当前 docstring 中 `source_positions[source site]` 的说法错误。

修正文档，使以下三种索引清晰区分：

- canonical algebra site；
- canonical order 中对应的 source fractional position；
- 原始 source `Atoms` 的 atom index，即 `atom_map[canonical_site]`。

保留 `e94e493` 的代码修复及 atom permutation 回归测试。

## 4. JAX 与环境清理

生产代码已经使用 Numba，但仓库仍存在会造成误解的 JAX 痕迹。合并前完成以下清理。

### 4.1 生产依赖

断言以下范围没有 JAX 导入或依赖：

```bash
rg -n 'jax|jaxlib|jnp|xla' src tests pyproject.toml --glob '*.py' --glob '*.toml'
```

允许 CHANGELOG 用文字说明 JAX 已删除；MathJax 是网页公式渲染器，与 Python JAX 无关。

### 4.2 `uv.lock`

仓库已忽略 `uv.lock`，主工作区可能残留迁移前生成的本地锁文件。它不是分支内容，但使用 `uv --frozen` 时可能安装旧 JAX。

在每个测试 worktree 中重新生成本地锁并验证：

```bash
uv lock
uv sync --group dev
uv run python -c "import importlib.util; assert importlib.util.find_spec('jax') is None"
```

不要把 `uv.lock` 重新纳入提交，除非项目单独改变锁文件政策。

### 4.3 研究原型

`research/ase_calculator/prototype.py` 仍直接导入 JAX，但 JAX 已不是项目依赖。二选一：

1. 删除该已失效研究原型；或
2. 改写为当前 Numba/NumPy API，并增加最小可运行验证。

不要保留一个在声明环境中无法运行的已跟踪 Python 原型。worktree 中未跟踪的 `bench_numba.py`、旧 JSONL 和研究草稿不进入提交，也不要未经所有者许可删除。

### 4.4 教学日志

以下当前教学日志仍包含旧 JAX/CUDA fallback 输出：

- `tutorial/long-range-electrostatics/NaCl/fit.log`；
- `tutorial/scph/K4As4Pt2/fit.log`；
- `tutorial/Si/force-fitting-ase/fit.log`；
- `tutorial/sscha/K4As4Pt2/T300K/fit.log`。

如果对应任务仍是当前教学案例，必须在新代码上完整重跑并覆盖日志。不得手工删掉 JAX 行。每个任务脚本仍必须独立捕获完整 stdout、stderr 和 traceback 到本目录固定 `fit.log`。

## 5. API 破坏策略

本轮不做兼容，按以下规则收口：

- 不恢复 `basis`、`pivots` 或旧 `symmetry` 字段；
- 不为旧 dataclass 构造签名提供默认值；
- 不保留 `cutoff=None`；
- 不保留 `regularization="scaled_group_lasso"`；
- 不接受旧裸 force arrays 进入 `reap()`；
- 不读取旧参数向量并猜测它属于 pivot 基还是 $Q$ 基；
- 如果存在持久化参数或 plan 文件，增加 schema version，并只读取新 schema；
- 所有旧输入得到面向用户的明确异常，不使用 `AttributeError`、底层 NumPy shape error 或错误物理结果作为“破坏”。

破坏性变更仍然需要设计良好的失败边界。所谓“不兼容”是拒绝旧语义，不是让旧调用随机失败。

## 6. 推荐提交结构

在现有 `reflector` 六个提交之后追加，不重写已经审阅过的核心提交：

1. `api: require explicit interaction cutoffs`
   - 删除 `cutoff=None`；
   - 更新测试、教程参数和中英文文档。
2. `fit: remove basis-dependent scaled group lasso`
   - 删除 ADMM、结果字段、测试和文档入口。
3. `finite-difference: bind force batches to a canonical plan`
   - manifest、fingerprint、新 `sow/reap` 数据对象和拒绝旧输入测试。
4. `algebra: canonicalize exact kernel and observation choices`
   - HNF 或明确的核格指纹；
   - $QO$ observation invariance 测试；
   - 修正文档声明。
5. `chore: remove stale jax research and regenerate teaching logs`
   - 删除或迁移研究原型；
   - 完整覆盖重跑受影响教学任务。
6. `docs: record the intentional 4.0 development break`
   - CHANGELOG、API、finite-difference 外部工作流和迁移说明。

每个提交必须独立通过与其范围对应的聚焦测试。不要使用一个大提交同时隐藏 API 删除、算法修复和生成产物。

## 7. 必须增加的测试

### 7.1 规范 frame

- cubic、fcc、diamond、hexagonal、rhombohedral、倾斜多原子；
- determinant 为 $1$ 和 $-1$ 的 unimodular 变换；
- 多次随机 elementary shear；
- source atom permutation；
- wrapped/unwrapped fractional positions；
- algebra cell、整数变换、motif mapping 的规范结果。

### 7.2 精确代数

- 秩与 SymPy oracle 一致；
- 大于 `int64` 的输入；
- 多个坏素数；
- kernel 精确、满维且饱和；
- 随机 kernel-basis unimodular 重参数化后的规范结果一致；
- integer-range failure 使用专用异常。

### 7.3 物理不变性

- 二至四阶 orbit 数和维数；
- Cartesian projector；
- expansion 后的物理 cluster/tensor；
- fitting 的无正则唯一解；
- ASR 约束后的 tensor；
- finite-difference reconstruction；
- realization identifiability；
- large shear 下不溢出。

### 7.4 计划身份

- 新 manifest 往返；
- 配置乱序；
- 缺失、多余和重复 ID；
- fingerprint 不符；
- 相同配置数但不同 observation rows；
- 旧裸数组与旧 mapping 明确失败。

### 7.5 已删除 API

为有意删除增加负面测试：

- `cutoff=None` 被拒绝；
- `regularization="scaled_group_lasso"` 被拒绝；
- 旧 `basis/pivots` 不存在；
- 旧 `reap(ndarray)` 被拒绝；
- 错误消息指向新 API，而不是建议兼容开关。

## 8. 测试环境与命令

先确保 worktree 使用新解析的无 JAX 环境。然后运行：

```bash
uv run pytest tests/test_lattice_frame.py
uv run pytest tests/test_exact_rank_kernel.py
uv run pytest tests/test_orbit_characterization.py
uv run pytest tests/test_orbit_invariance.py
uv run pytest tests/test_core_real_space.py
uv run pytest tests/test_reconstruction_solver.py
uv run pytest tests/test_api_sow_reap.py
uv run pytest tests/test_fitting_model.py tests/test_fitting_constraints.py
uv run pytest tests/test_documentation_sync.py tests/test_logging_contract.py
uv run ruff check src tests
```

对本分支修改过的 Python 文件运行格式检查：

```bash
git diff --name-only 763544c...HEAD | rg '\.py$' | xargs uv run ruff format --check
```

完整测试依赖 `phonopy` 和 `phono3py` oracle。项目必须二选一：

- 把它们放入明确的 reference-test dependency group，并在该环境运行全套；或
- 将相关测试按可选 oracle 正确 skip，不能在 collection 阶段直接失败。

既有 `fitting -> calculators` 架构 DAG 失败也必须在最终合并报告中单列。最理想是先修复；若不属于本分支，至少要在 `dev` 和候选分支使用同一环境逐项证明结果相同。

## 9. 教学任务验收

枚举所有独立拟合入口，而不是只更新本次 diff 中已经变化的 13 个目录：

```bash
rg -l 'ForceConstantFitter|FiniteDifferenceCalculation|LoopSCPH|SSCHA' tutorial --glob '*.py'
```

逐项判断是否消费了本轮改变的 cutoff、参数基、有限差分计划或拟合器。受影响任务必须完整重跑：

1. 脚本以覆盖模式创建本目录 `fit.log`；
2. stdout、stderr 和 traceback 全部进入日志；
3. `fit.log` 不被 `.gitignore` 排除；
4. `metrics.json` 与日志来自同一次运行；
5. 日志不再出现旧 JAX runtime 输出；
6. 显式 cutoff、orbit 数、参数数、误差与约束残差均被记录；
7. 失败任务提交完整 traceback，修复后重新覆盖，不手工编辑日志。

## 10. 最终合并门槛

只有全部满足时才能合并 `reflector`：

- primitive interaction space 不再接受 `cutoff=None`；
- 当前 basis-dependent scaled group LASSO/ADMM 已删除；
- 旧 finite-difference forces 不可能被新计划静默接收；
- 新计划具有稳定 manifest、schema 和 fingerprint；
- 精确核的规范性承诺已经明确并由属性测试证明；
- observation rows 对 $QO$、unimodular cell 和 atom permutation 不变；
- 同一显式 primitive 模型不依赖 reference 选择；
- Cartesian projector、展开 IFC、拟合预测和有限差分结果保持物理不变；
- 生产源码、测试环境和当前教学日志不再依赖 JAX；
- 已跟踪研究 Python 文件在声明环境中可运行，或已删除；
- 所有受影响教学任务的 `fit.log` 已完整覆盖生成；
- 聚焦测试、文档测试、日志测试和 Ruff 全绿；
- 完整测试在声明的 reference-test 环境中通过，或仅剩逐项证明为 `dev` 既有且经明确批准的失败；
- CHANGELOG 明确登记所有破坏性变更，不提供兼容别名或静默回退。

## 11. 交付报告格式

完成后提交以下信息供最终审阅：

1. 从 `763544c` 到候选 HEAD 的提交列表；
2. 所有已删除 API 的清单；
3. finite-difference manifest 示例及 fingerprint 构成；
4. 旧 forces 被拒绝的测试输出；
5. kernel 规范化属性测试结果；
6. unimodular/atom-order/正交换基属性测试结果；
7. 最大 observation condition；
8. 全套测试、Ruff、文档和日志检查结果；
9. 所有重跑教学任务及其 `fit.log` 列表；
10. `rg` 证明生产代码无 JAX 的结果；
11. worktree 中未跟踪文件列表，确认它们没有进入提交。

最终审阅只依据已提交树和可重复命令，不以未跟踪 benchmark、手工说明或某次交互会话作为正确性证据。
