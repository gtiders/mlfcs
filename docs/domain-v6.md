# 版本 6 域与执行契约

最初的[数学审查](numba-integer-audit.md)与[迁移研究](unified-numba-backend.md)作为历史证据保留。本页记录其后批准的对象合并，并取代两者的 API 示例。

## 直接构造与显式几何

```python
import numpy as np
from ase.build import bulk
from mlfcs import ClusterMap, ClusterSpace, ForceDataset, FitSystem

primitive_atoms = bulk("Si", "diamond", a=5.43)
cs = ClusterSpace(primitive_atoms, cutoffs={2: 4.0, 3: 4.0})
csmap = ClusterMap(
    cs,
    primitive_atoms.repeat((3, 3, 3)),
    supercell_matrix=3 * np.eye(3, dtype=np.int64),
)
csmap.rank_info().require_full()
# system = FitSystem(ForceDataset(csmap, evaluated_structures))
```

阶即 `cutoffs` 的键。每个最大体序默认等于其张量阶。几何验证、spglib 预处理、阶段局部整数准入、簇枚举与不变量参数化都在 `ClusterSpace` 初始化内完成。`ClusterMap` 初始化验证 supercell 几何、准备周期 quotient 表并折叠轨道像。

`cs.primitive_atoms` 与 `csmap.supercell_atoms` 返回脱离的 ASE 快照，含原子质量。域数组连续且只读。不存在含义模糊的 `.atoms` 字段。折叠轨道的原子索引是 `csmap.image_atom_indices`；原子寻址使用 `csmap.atom_index(lattice_site)`。

`cs.masses` 暴露 readonly 的原胞位点质量。`cs.with_masses(values)` 在共享几何、对称与轨道数据的前提下创建新的质量赋值，不重建模型空间。用 `ForceConstants(new_cs, model.coefficients)` 重绑系数；见[谐波频率与原子质量](harmonic-api.md)。

`PrimitiveCell` 与 `Supercell` 已移除。不提供任何几何 `build`、`from_atoms`、`map_to`、`.primitive`、`.supercell` 或 `.space` 兼容别名。Taylor calculator 支持已移除。`ForceDataset` 收集已求值的训练帧；`FitSystem` 从数据集构造方程。

## 所有权与依赖方向

```text
ASE primitive atoms
    -> ClusterSpace (owns primitive arrays, symmetry, orbits)
        -> PrimitiveSymmetry (array representation, no ASE object)

ClusterSpace + ASE supercell atoms + integer supercell matrix
    -> ClusterMap (references ClusterSpace; owns supercell and folding arrays)
        -> ForceDesign (references ClusterMap; owns reusable compiled design)
        -> FiniteDifference (references ClusterMap)
        -> FitSystem (references ClusterSpace; owns raw equations or normal statistics)
            -> ForceConstants (references ClusterSpace; owns coefficients)
```

一个 `ClusterSpace` 可以支撑多个独立初始化的 `ClusterMap`。不引入 `MappedClusterSpace` 或 `ClusterModel` 包装。原胞模型与保存的力常数都不强制拥有某个 supercell，也不保留训练快照。

## 语义模块布局

| 模块 | 职责 | 执行 |
|---|---|---|
| `_arrays`, `errors` | 准入、连续不可变数组、错误类型 | Python 边界 |
| `math` | 跨领域整数矩阵规范化、3×3 格点运算与模消元原语 | Python + Numba |
| `core/structure`, `symmetry` | 验证数组、坐标约定、spglib 作用 | Python 预处理 |
| `core/tensors` | 共享张量作用与笛卡尔收缩 | Numba + NumPy |
| `cluster_space/candidates`, `orbits`, `basis`, `integer_kernel` | 邻居、候选、轨道作用、不变量基与饱和整数核 | Python 编排 + Numba |
| `mapping/supercell`, `periodic`, `cluster_map`, `folding` | 显式 supercell 匹配、quotient 寻址、折叠 rank | Python 准入 + Numba 循环 |
| `fitting/design`, `system`, `solve` | 可复用力设计与每线程临时数组 | Python 准备 + 并行 Numba 累加 |
| `force_constants`, `finite_difference` | 系数、约束、导出与力采样 | Python 编排 |
| `phonon/grid` | 倒格点网格与倒格标签的整数变换 | Python 验证 + Numba 循环 |

折叠 rank 的认证流程属于 `mapping/folding`，并与整数核算法共享 `math.echelon` 和素数流。不变量构造调用 `integer_kernel_basis`；有理 chart 与同余原像确定饱和格基。倒格点整数乘法和幺模逆由 `phonon/grid` 内部实现，不作为共享矩阵 API。运算命名描述这些数学对象；其定义给出系数域、输入范围与失败条件。

`Orbit.lattice_basis` 在格坐标下生成 stabilizer 不变格。`component_basis` 把物理参数映射到笛卡尔张量分量。`ClusterMap.rank_info()` 报告折叠后参数列的有理数域秩；`integer_kernel_basis(A)` 返回生成 `A @ x == 0` 全部整数解的列。倒格点标签按其定义的对偶晶格作用变换，并在需要时按网格分母取模。这些定义取代先前 `exact_*` 与 `certified_*` 接口限定词；旧的导入路径与名称已移除。

## 准入、数组 ABI 与工作区

每个编译入口消费连续的 `int64`/`float64` 数组与标量元数据。NumPy 分配的维度与字节长度对照 `np.intp` 检查；数学整数使用对称 int64 域。准入检查在需要它的运算之前执行；域对象不保留、不传递 certificate 记录。

局部证明在每个运算之前就其实际数据执行。整阶参数、image 与张量存储上界已被移除，连同其准入分支。sizing 趟在超过实际输出容量之前停止；填充与张量循环使用已准入的数据，不做逐运算溢出检查。周期 quotient 寻址改为对实际加法、乘积与部分和使用单个受检 Numba 运算，避免聚合筛查与独立回退。输入验证、实例特定的模重构与同余准入、精确 certificate、形状检查与工作区线程数检查仍然适用。通过几何证明不意味着每个任意精确矩阵都被准入。

数值消费者直接使用不可变域数组。力设计持有变换后的 image 基；它们是派生数据。工作区属于一次设计调用，具有显式线程容量，不得并发使用。并行累加使用独立的每线程数组再归约。JIT 特化跟随数组 dtype/布局；张量阶保持运行时值。

## 原生模型存储

原生存储使用 HDF5 格式版本 5，轨道格生成元存储为 `lattice_basis`。只有 ForceConstants 暴露 save/load。几何、质量、对称、块、轨道基与系数都是显式数值数据集与属性；没有对象编码，也没有内容指纹。更旧的原生文件被拒绝。

加载验证数组类型、形状、有限值、质量与模型布局，不重复邻居或轨道枚举。JIT 缓存、工作区、map 与拟合系统不被持久化。本地教学模型与参考 fixture 已一次性转换，几何、基与系数条目全部不变。独立提供的模型与映射之间的兼容性由调用者负责；merge、导出与 SCPH 不比较模型身份。

原胞初始化使用 ASE get_scaled_positions(wrap=True)，保留输入 Atoms 对象与原子顺序。数组验证要求分数坐标在 $[0, 1)$；加载检查存储坐标时不再次 wrap。

## 验证

冻结约束矩阵、lattice oracle、轨道/作用记录、原 Python 力设计快照、多 supercell 折叠测试与教学流程仍是数值参考。测试覆盖正常计算、快照隔离、质量保持、只读恢复与物理数值结果。架构门禁、被移除接口的断言与故意非法输入测试已被排除。可选的科学 oracle 对照使用 `reference` 标记。教学拟合的完整结果记录由其教程 notebook 的提交输出承担。

证明、所有权与清理规则见[本地整数契约与清理规则](local-integer-contracts.md)。

## 流式 ASR 投影

平移不变性统一由 `force_constants.asr` 实现。`AcousticSumRuleOperator` 向 LSMR 提供 $v\mapsto Av$ 与 $u\mapsto A^Tu$，无需构造高阶显式矩阵。`acoustic_constraint_matrix` 表示相同的方程，服务于 FC2 旋转投影。方程固定前 $p-1$ 个原子位置和全部 Cartesian 方向，对最后一个位置及其周期镜像求和；列对应当前阶的物理分量参数。重复贡献先按符号合并，再计算残差尺度。

`enforce_asr` 逐阶求满足 $Ac'=0$ 的最小欧氏参数修正。`rtol` 控制最大绝对方程残差相对于 $\|A\|_\infty\|c'\|_\infty$ 的比例，而非相对于初始残差。当前 ASR 使用浮点 LSMR；整数核用于更早的轨道 stabilizer 不变基构造，不用于这一步投影。

对照验证（Si FC2–FC5，rtol=1e-10）：流式与稀疏路径的 LSMR 迭代数逐阶相同（1/18/36/74），投影系数最大绝对差 4.7e-12（FC5），`relative_before` 逐位一致；算子构建 0.08 s（稀疏构建 39.5 s）。流式路径的常驻内存全程约 0.45 GiB；对照观测到的 5.5 GiB 峰值属于稀疏 oracle 自身的构建。对照为一次性验证，常驻测试覆盖 `enforce_asr` 端到端行为（残差阈值、幂等性、物理残差交叉核对）。

## 旋转与平衡条件

`force_constants.rotation` 构造并投影 FC2 的距离矩。Born–Huang 条件使用实际原子间距的一次矩，采用原子处于力平衡时的齐次形式；Huang 条件使用二次矩，并要求零应力，因此默认关闭。[条件定义见 Lin、Poncé 与 Marzari 的式 (6)、(16)](https://arxiv.org/html/2209.09520v2)。一般高阶旋转条件联系相邻阶力常数，当前接口不将其独立逐阶投影。

令 $C$ 为除以相应长度尺度后的旋转方程，$W$ 为声学矩阵行空间的正交基。旋转修正通过有效矩阵 $E=C-(CW^T)W$ 的 SVD 求解，因此满足 $A\Delta c\approx0$，保留已有声学残差。需要两种约束时，先调用 `enforce_asr()`，再对返回的模型调用 `enforce_rotation()`。两步都在物理分量参数坐标下最小化改变量，不代表最小化全部展开张量元素的改变量。

`rank_rtol` 是最大奇异值的截断比例，不是残差容差。默认比例为实际位置对称误差除以中位非 onsite 距离，再加 Cartesian 旋转的非正交误差，取和的两倍；始终保留机器精度下限。被截断的方向不予修正，报告可能保留相应残差。Born–Huang 和 Huang 的绝对残差分别以 eV/Å 和 eV 表示，声学残差为 eV/Å²。

整数 lattice rotation 不会使实际距离矩自动成为整数。通用输入的内部坐标没有有界小分母保证，本实现不为整数化而近似几何。即便特殊结构允许整数核，精确生成元也不保证良好条件数；转换到物理参数后仍需正交化并保持原投影度量。两个后处理模块继续使用现有 LSMR/SVD 算法，不引入整数投影分支。
