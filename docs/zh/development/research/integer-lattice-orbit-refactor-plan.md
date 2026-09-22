# 整数晶格轨道代数重构：Git 审阅、实施与验证指南

## 1. 目标与范围

本文供接手开发者审阅并重新实现以下四个远程提交中的有效设计：

| 提交 | 主题 | 初步处置 |
| --- | --- | --- |
| `ad56509` | 在整数晶格参考系中进行轨道代数 | 不原样合并，拆分重构 |
| `36a89d5` | 使用双向模素数证书计算精确秩 | 修复整数表示后独立移植 |
| `83a6510` | 按需生成秩证书使用的素数 | 保留思路，修正文档和边界 |
| `b6f7802` | 使用 SymPy 生成证书素数 | 在明确整数契约后移植 |

本任务不是简单地连续 `cherry-pick` 四个提交。目标是保留“在整数晶格坐标中进行精确对称代数”的正确方向，同时修复以下结构问题：

1. 原始 primitive cell 未经规约，等价的 unimodular 基变换会导致整数系数和高阶张量迅速膨胀。
2. 精确整数基、笛卡尔物理子空间和数值拟合基被混为一个 `basis`。
3. `pivots` 已经不再代表单位 pivot 参数，而只是选中的笛卡尔观测行。
4. 所谓精确整数链路仍强制使用 `int64`，合法的大整数会溢出。
5. `ad56509` 同时删除了 `cutoff=None`、group LASSO/ADMM 等无关功能，不应随坐标重构一起进入主线。

最终实现必须保持既有公共行为，除非另有独立、经过批准的 API 变更。

## 2. 仓库规则

开始前先阅读仓库根目录的 `AGENTS.md`。本任务尤其要遵守：

- Markdown 数学公式只使用 `$...$` 和 `$$...$$`。
- 如果修改或新增教学拟合任务，任务脚本必须把完整 stdout、stderr 和 traceback 覆盖写入任务目录固定的 `fit.log`；该日志必须纳入 Git，且不能被 `.gitignore` 排除。
- 不修改、暂存或删除其他开发者尚未提交的文件。
- 不用 `git reset --hard`、`git checkout -- <path>` 或 `git clean` 清理工作区。

## 3. 先确认提交拓扑与工作区

在现有仓库中只执行只读命令：

```bash
git status --short --branch
git remote -v
git fetch origin
git log --oneline --decorate --graph --max-count=30 --all
git merge-base dev origin/numba-fitting
git log --reverse --oneline dev..origin/numba-fitting
```

截至本指南编写时，相关远程序列是：

```text
ad56509 decide orbit algebra in the integer lattice frame
36a89d5 certify the exact rank with primes in both directions
83a6510 stream rank certificate primes without a cap
b6f7802 take certificate primes from sympy instead of a hand-written test
```

当前 `dev` 已经用不同提交哈希安全移植了更早的文档清理、Numba 拟合、最小镜像和输入检查修复。因此不要用“分支 ahead/behind 数量”判断哪些功能缺失，应按文件内容和行为比较。

## 4. 使用独立 worktree

当前主工作区可能存在未提交文件。使用两个独立 worktree：一个只用于观察远程最终状态，一个用于真正实现。

```bash
git worktree add --detach ../mlfcs-upstream-lattice-review b6f7802
git worktree add -b refactor/integer-lattice-frame ../mlfcs-integer-lattice dev
```

两个目录的用途如下：

- `../mlfcs-upstream-lattice-review`：运行远程版本、查看实现和复现问题，不提交修改。
- `../mlfcs-integer-lattice`：从当前 `dev` 开始分阶段实现，每个阶段独立提交。

若分支名或目录已存在，先用以下只读命令确认来源，不要直接删除：

```bash
git worktree list
git branch --list 'refactor/integer-lattice-frame'
```

## 5. 如何逐个查看远程修改

### 5.1 查看提交摘要和文件范围

```bash
git show --stat --summary ad56509
git show --stat --summary 36a89d5
git show --stat --summary 83a6510
git show --stat --summary b6f7802

git diff-tree --no-commit-id --name-status -r ad56509
git diff-tree --no-commit-id --name-status -r 36a89d5
git diff-tree --no-commit-id --name-status -r 83a6510
git diff-tree --no-commit-id --name-status -r b6f7802
```

### 5.2 查看单个提交的实际补丁

```bash
git show --find-renames --find-copies ad56509
git show --find-renames --find-copies 36a89d5
git show --find-renames --find-copies 83a6510
git show --find-renames --find-copies b6f7802
```

若只看核心实现：

```bash
git show ad56509 -- src/mlfcs/interactions
git show ad56509 -- src/mlfcs/finite_difference src/mlfcs/fitting
git show 36a89d5 -- src/mlfcs/interactions/algebra/exact.py tests/test_core_real_space.py
git show 83a6510 -- src/mlfcs/interactions/algebra/exact.py tests/test_core_real_space.py
git show b6f7802 -- src/mlfcs/interactions/algebra/exact.py tests/test_core_real_space.py
```

### 5.3 查看四个提交叠加后的最终状态

```bash
git diff ad56509^..b6f7802 -- src/mlfcs tests
git diff --stat ad56509^..b6f7802
git grep -n 'pivots\|scaled_rotation\|certified_rank\|invariant_kernel' b6f7802 -- src tests
```

### 5.4 比较当前 `dev` 与远程结果

不要直接比较提交哈希，应比较目标文件：

```bash
git diff dev..b6f7802 -- src/mlfcs/interactions
git diff dev..b6f7802 -- src/mlfcs/finite_difference src/mlfcs/fitting
git diff dev..b6f7802 -- tests/test_core_real_space.py
```

还应检查 `ad56509` 中与本任务无关的删除：

```bash
git diff ad56509^..ad56509 -- src/mlfcs/fitting/linear_solvers.py
git diff ad56509^..ad56509 -- src/mlfcs/fitting/fitter.py
git diff ad56509^..ad56509 -- src/mlfcs/interactions/space.py
```

任何涉及 group LASSO/ADMM、`cutoff=None`、教程结果更新或其他公共 API 删除的变化都先排除出本重构。

## 6. 先画清当前数据流

修改代码前，沿以下顺序阅读并记录对象的输入、输出、坐标系、dtype 和形状：

1. `src/mlfcs/interactions/primitive/builder.py`
2. `src/mlfcs/interactions/primitive/candidates.py`
3. `src/mlfcs/interactions/algebra/actions.py`
4. `src/mlfcs/interactions/algebra/invariants.py`
5. `src/mlfcs/interactions/algebra/indexed_orbit.py`
6. `src/mlfcs/interactions/models.py`
7. `src/mlfcs/interactions/realization.py`
8. `src/mlfcs/fitting/parameterization.py`
9. `src/mlfcs/finite_difference/reconstruction.py`
10. `src/mlfcs/force_constants/expansion.py`
11. `src/mlfcs/fitting/constraints.py`
12. `src/mlfcs/constraints/translational.py`

对每个中间量至少回答：

- 原子位置是用户 cell、规约 cell 还是笛卡尔坐标？
- 平移向量属于哪个整数晶格基？
- 旋转矩阵作用于 direct fractional 分量还是 Cartesian 分量？
- 张量基的列表示精确晶格系数、Cartesian 分量还是正交数值参数？
- 一个行索引指向哪个坐标系中的张量分量？
- 该对象是否允许浮点误差？是否允许超出 `int64`？

建议先写出并验证以下约定：若 direct lattice cell 记为 $A$，则 $n$ 阶张量从晶格分量到笛卡尔分量的映射为

$$
K_n=(A^\mathsf{T})^{\otimes n}.
$$

不要只依据变量名推断转置方向，应使用非正交晶胞和单位张量分量做数值测试。

## 7. 目标架构

### 7.1 `LatticeFrame`

在 `ReferenceFrame` 附近引入一个明确的晶格参考系对象，至少包含：

- `source_cell`：用户输入晶胞；
- `algebra_cell`：规约后用于整数代数的晶胞；
- `source_to_algebra`：精确整数 unimodular 矩阵 $U$；
- `algebra_to_source`：$U^{-1}$，同样必须是精确整数；
- 原子分数坐标、整数平移、空间群操作在两个参考系间的转换；
- 按阶生成 $K_n$ 的唯一入口。

必须验证：

$$
\det U=\pm1,
$$

并验证 `source_cell` 与 `algebra_cell` 通过 $U$ 表示同一 Cartesian 晶格。不能静默丢弃用户原始晶胞。

### 7.2 规约 primitive lattice

在生成候选 cluster 和张量群作用之前，对 primitive lattice 做 Minkowski reduction 或等价规约。规约必须满足：

- 返回精确整数变换；
- 原子 motif 能正确映射并重新匹配；
- 周期平移保持整数；
- 对符号、轴置换和等长简并给出确定性结果。

如果 ASE 的规约对简并基不能保证唯一，应在有限个 signed permutation 候选中使用稳定键选择规范表示。稳定键必须包含 motif 信息，避免只看 cell 而错误交换不等价原子。

### 7.3 每个轨道保存两套基

令精确整数不变量基为 $B_{\mathbb Z}$，Cartesian 物理张量子空间为

$$
C=K_n B_{\mathbb Z}.
$$

对 $C$ 做稳定 QR 或 SVD：

$$
C=Q R.
$$

用途必须分离：

- `exact_lattice_basis`：$B_{\mathbb Z}$，只用于精确群代数、秩、可辨识性和来源追踪；
- `cartesian_basis`：$Q$，用于拟合、有限差分重建、约束和力常数展开；
- `coefficient_transform`：$R$，用于精确系数 $c$ 与数值系数 $\theta=Rc$ 之间的转换。

不要再用一个含混的 `basis` 同时承担三种角色。

### 7.4 将 pivot 改为观测行

在 $Q$ 上用 rank-revealing QR 或 max-volume 方法选择观测行，保存：

- `observation_rows`；
- `observation_matrix = Q[observation_rows, :]`；
- 观测矩阵的条件数。

除非满足 `basis[pivots] == I`，否则不能再把这些行叫作参数 pivot。有限差分重建应显式求解

$$
Q_{\mathrm{obs}}\theta=y_{\mathrm{obs}},
$$

并使用稳定求解器，而不是隐含假设观测值等于参数。

### 7.5 集中 Cartesian 渲染

建立一个统一入口把轨道基渲染到 Cartesian 空间。拟合、展开、ASR、有限差分和其他约束均调用该入口，不应各自散落 `frame @ basis`。

这样可以在一个位置检查：

- 张量阶数；
- 坐标系；
- 列数和参数切片；
- 精确基到数值基的变换；
- dtype 和缓存生命周期。

## 8. 精确秩与整数核的实施要求

### 8.1 不要无条件转成 `int64`

输入首先应保持 Python arbitrary-size integer 或 SymPy integer。若要使用快速 `int64` 模运算，必须先逐项计算 $a_{ij}\bmod p$，因为模 $p<2^{31}$ 的余数才安全进入 `int64`。

如果项目选择只支持 `int64`，也必须在入口进行显式、可读的范围检查，不能让 NumPy 或 C 扩展在中途抛出含混的转换错误。

### 8.2 秩证书

可以保留 `36a89d5` 的证明框架：

- 模 $p$ 的秩给出有理数秩的下界；
- Hadamard 上界和已尝试素数乘积给出上界证书；
- 某个素数上的满秩可立即证明有理数域满秩。

但实现需要：

- 对大整数安全计算 Hadamard 上界；
- 明确零维和空矩阵语义；
- 记录使用过的不同素数，不能重复计算乘积；
- 当素数范围耗尽时抛出专用异常；
- 文档称其为“按需素数流”，不要声称数学意义上的无限流。

使用 SymPy `prevprime` 替代自写素性测试是合理的，但应当只是素数来源，不应让 SymPy 隐式决定其他数据类型或异常语义。

### 8.3 不变量核

当前“固定少量素数选 pivot、浮点求解、分数重建、最后精确验证”的实现通常能拒绝错误结果，但会对合法的病态表示失败。

优先选择：

1. 直接对堆叠的稳定子残差约束计算模秩和核；
2. 使用 SymPy `DomainMatrix`、HNF 或 SNF 构造有理核及其饱和整数格；
3. 只在经过明确范围检查后启用 `int64` 快速路径；
4. 最终总是精确验证约束矩阵乘以整数基为零。

逐列除以最大公约数不足以证明整个列格已经饱和，必须补充饱和性或规范形验证。

尽量避免先构造 Gram 矩阵 $M^\mathsf{T}M$：它会平方放大整数，而且在有限域中可能产生额外退化。若因性能保留 Gram 路径，应提供直接约束路径作为可靠回退。

## 9. 分阶段实现与提交

每个阶段单独提交，阶段间保持测试通过。

### 阶段 A：测试刻画，不改变行为

新增测试描述现有坐标约定、orbit 参数维数和 Cartesian 子空间。先让测试在当前 `dev` 上通过。

建议提交信息：

```text
test: characterize orbit frame and basis semantics
```

### 阶段 B：引入 `LatticeFrame`

实现 source/algebra cell、精确整数变换和往返映射。此阶段不切换生产数据流。

```text
refactor: introduce explicit lattice algebra frame
```

### 阶段 C：规约后进行轨道代数

把候选、平移和群作用切换到规约参考系，保留源晶胞映射。确认轨道数、维数和 Cartesian 子空间不变。

```text
refactor: build orbit algebra in a reduced lattice frame
```

### 阶段 D：分离精确基与数值基

加入 $B_{\mathbb Z}$、$Q$ 和 $R$，迁移拟合、重建、展开与约束。此阶段完成 `pivots` 重命名。

```text
refactor: separate exact and numerical orbit bases
```

### 阶段 E：精确秩与整数核

选择性移植后三个提交的秩证书逻辑，补 arbitrary-integer 支持和可靠核构造。

```text
feat: certify exact ranks with modular arithmetic
```

### 阶段 F：文档与兼容层

更新中英文理论和 API 文档，纠正旧的 `normalize_pivot_basis`、fraction-free elimination 和 unbounded prime stream 等描述。若需要保留旧字段，应提供带弃用提示的只读兼容属性。

```text
docs: explain lattice and Cartesian orbit bases
```

`cutoff=None`、group LASSO/ADMM 或其他功能删除必须另开议题和提交，不属于上述任何阶段。

## 10. 必须新增的验证

### 10.1 Unimodular 不变性属性测试

对同一物理结构生成一组 $U\in GL(3,\mathbb Z)$，包括较大的 shear。至少测试二、三、四阶：

- 轨道数量一致；
- 每个轨道的不变量维数一致；
- Cartesian 子空间投影一致；
- 展开得到的力和能量一致；
- 有限差分重建一致；
- realization identifiability 的秩和结论一致；
- 不发生整数溢出。

比较子空间时不要逐列比较基，因为基可以相差可逆变换。若 $Q_1,Q_2$ 是正交基，应比较

$$
Q_1Q_1^\mathsf{T}
\quad\text{与}\quad
Q_2Q_2^\mathsf{T}.
$$

### 10.2 非正交材料

至少包含：

- 简单立方单原子；
- diamond/FCC primitive cell；
- 六方双原子结构；
- 菱方或明显倾斜的 primitive cell；
- 带多个同种或不同种原子的 motif。

### 10.3 条件数

记录每个轨道的 `observation_matrix` 条件数。对于等价 unimodular 表示，条件数不应随原始 basis shear 出现多个数量级的恶化。测试应覆盖四阶，因为问题会随张量阶数放大。

### 10.4 精确秩

使用 SymPy 作为小矩阵 oracle，固定随机种子验证：

- 零矩阵、空矩阵、满秩和秩亏矩阵；
- 重复行列；
- 恰好使若干小素数成为坏素数的矩阵；
- 接近 `int64` 边界的整数；
- $2^{63}$、$10^{30}$ 等大整数；
- 稀疏和高动态范围矩阵。

如果决定不支持 arbitrary integer，最后两类必须得到项目自定义的明确范围异常，而不是 `OverflowError` 或静默截断。

### 10.5 坐标往返

分别测试：

- source fractional 到 algebra fractional 再返回；
- source translation 到 algebra translation 再返回；
- 空间群旋转的共轭变换；
- 二、三、四阶张量的 lattice/Cartesian 往返；
- 原子重排和周期 wrap 后的 motif 对应。

## 11. 运行测试和静态检查

在实现 worktree 中创建或同步开发环境：

```bash
uv sync --group dev
```

先运行聚焦测试：

```bash
uv run pytest tests/test_integer_lattice.py
uv run pytest tests/test_core_real_space.py
uv run pytest tests/test_reconstruction_solver.py
uv run pytest tests/test_fitting_model.py tests/test_fitting_constraints.py
uv run pytest tests/test_core_symmetry.py
```

再运行完整检查：

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

若本分支没有 `examples/`，先用 `git ls-files examples` 确认，再将不存在的目录从 Ruff 命令中移除，不要创建空目录来迁就命令。

运行文档同步测试，其中应包含 Markdown 数学分隔符检查：

```bash
uv run pytest tests/test_documentation_sync.py
```

若该测试尚未覆盖仓库规则，应先扩展测试，使其拒绝非美元符号数学定界符，再重新运行。

如果改动教学拟合任务，还必须：

1. 从任务目录执行其拟合脚本；
2. 确认脚本以覆盖方式写 `fit.log`；
3. 检查 stdout、stderr 和 traceback 都进入日志；
4. 执行 `git check-ignore -v <任务目录>/fit.log`，该命令不应显示忽略规则；
5. 将更新后的 `fit.log` 和确定性结果一起提交。

## 12. 如何与远程实现做行为对照

不要让两个 worktree 共用同一个可编辑虚拟环境。分别在两个目录运行相同的最小脚本，并保存：

- orbit 数量和各 orbit 维数；
- $B_{\mathbb Z}$、$Q$ 的形状；
- Cartesian projector；
- 观测行和条件数；
- 精确秩；
- 构造时间和峰值内存；
- 所有异常及完整 traceback。

对照结果时，将旧远程实现视为“设计参考和反例对象”，不是正确性 oracle。正确性 oracle 应来自：

- 群作用与坐标变换的数学恒等式；
- SymPy 小规模精确运算；
- 等价晶胞表示的不变性；
- 已有公共 API 回归测试；
- 直接 Cartesian 枚举的小体系结果。

## 13. 每阶段提交前的 Git 检查

```bash
git status --short
git diff --check
git diff --stat
git diff -- src tests docs
git diff --cached --stat
```

只暂存本阶段明确修改的路径：

```bash
git add path/to/file1 path/to/file2
git diff --cached
git commit
```

禁止使用 `git add -A`，以免把他人的临时文件、生成物或无关教程结果带入提交。

提交后检查边界：

```bash
git show --stat --summary HEAD
git show --check HEAD
git log --oneline --decorate dev..HEAD
```

## 14. 最终验收条件

只有同时满足以下条件，才建议把实现分支合并到 `dev`：

- 四个原提交没有被整体 cherry-pick；有效逻辑已按职责拆分。
- `cutoff=None`、group LASSO/ADMM 和其他无关公共能力没有被删除。
- 原始晶胞与规约代数晶胞之间存在显式、可验证的整数映射。
- 精确整数基与正交数值基分离，调用方不再猜测 `basis` 的坐标语义。
- `pivots` 已改为准确的观测行语义，重建显式使用观测矩阵。
- 大 unimodular shear 不再造成三阶构造溢出。
- 等价晶胞表示得到相同的 Cartesian 子空间和物理预测。
- 四阶观测矩阵的条件数受到控制，并有自动测试。
- 精确秩对大整数安全，或者具有明确、提前的范围错误。
- 中英文文档与最终实现一致，没有宣称不存在的“无限”或“任意精度”能力。
- 聚焦测试、完整测试、Ruff 和 Markdown 规则检查全部通过。
- 教学任务若有修改，其 `fit.log` 完整、覆盖生成且已纳入 Git。

## 15. 合并建议

开发分支应整理为数个可独立审阅的提交。合并前可用：

```bash
git log --reverse --oneline dev..refactor/integer-lattice-frame
git diff --stat dev...refactor/integer-lattice-frame
git diff --check dev...refactor/integer-lattice-frame
```

由维护者批准后，再在干净的集成 worktree 中执行非快进合并或逐提交 cherry-pick。不要在含有未提交文件的主工作区直接合并。

若某一阶段无法独立保持测试通过，应优先重新划分提交边界，而不是把所有修改压成一个大提交。
