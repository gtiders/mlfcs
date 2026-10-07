# 统一 Numba 后端

数学基线是 [Numba 迁移前数学审查](numba-integer-audit.md)。该报告原样保留。本文档描述实现契约。

## 域与依赖边界

`ClusterSpace` 持有不可变的原胞交互模型与 `PrimitiveSymmetry`。`ClusterMap` 持有一个显式 supercell 实现及其派生的 quotient、原子与折叠簇映射。一个 space 可以对应多个 map；space 与它的 map 都不保留训练结构。

```text
ASE/spglib → ClusterSpace ← ASE supercell atoms
                    │                │
                    └──────→ ClusterMap
                               ├── ForceDesign → FitSystem → ForceConstants
                               ├── FiniteDifference
                               └── reciprocal calculations
```

Python 负责验证、spglib 调用、编排与 NumPy/SciPy 线性代数。Numba kernel 位于拥有其输入的域运算旁边；它们接受连续数组与标量元数据，不接受 ASE 对象或 Python 容器。共享周期几何位于 `core/geometry.py`，在 Minkowski 约化之后按笛卡尔坐标求值。位点匹配在声明的容差内一次性枚举所有的像；拟合位移使用独立的最小像查询。

## 数组与安全契约

数学整数、标签、置换、索引与 offsets 使用 C 连续 `int64`；浮点数组使用 `float64`；掩码使用 `uint8`。被准入的整数区间排除 `INT64_MIN`，因此绝对值与符号归一化是安全的。形状与字节长度在分配之前对照 `np.intp` 检查。输入为只读；可写的调用者数组在归一化时复制。

几何入口先用截断/逆胞平移盒界定计数器。其邻居计数给出实际最大值 $M$，从而收紧候选上界 $N\binom{M+p-2}{p-1}$。后续的标签、张量、轨道像、参数与分配上界，都在这些值已知的阶段用 Python 整数求值。这些整数描述局部上界；它们不做特征零消元，也不随域对象传播。

热点循环使用 int64 算术。周期 quotient 映射在其单次 Numba 映射趟中检查每个实际的加法、乘积与部分和。分阶段的分母与分子检查先于各自的乘法执行。rank、重构与残差 certificate 仍是精确算法的一部分。

不存在 `PreparedClusterSpace` 或 `PreparedClusterMap` 包装。不可变域数组直接传给数值消费者。quotient 查找由 `ClusterMap` 持有，以有序数组加编译期二分查找实现。不存在隐式全局映射缓存。

## 整数核实现

`mlfcs.cluster_space.integer_kernel.integer_kernel_basis(A)` 返回 readonly 的 `int64` 列，生成 `A @ x == 0` 的全部整数解。signed 关联约束走 signed union-find 路径。一般矩阵走固定的双素数主元 chart、有理重构与复合同余原像。带残差上界的独立模零化 certificate 确立重构 chart 的上秩；非零主元子式确立其下秩。折叠秩认证由 `mlfcs.mapping.folding` 承担，模消元原语与整数核共享 `mlfcs.foundation.integer`。

复合步骤是特化的三角原像算法，而不是通用 Howell 库或复合模上的有限域 RREF。它产生列 HNF：正对角元、上三角，且 $0\le H_{ij}<H_{ii}$（$i<j$）。

对单行 $w$，令 $g_{-1}=\delta$、$g_j=\gcd(\delta,w_0,\ldots,w_j)$。第 $j$ 个对角主元是 $g_{j-1}/g_j$。一个有界的 Bezout 向量把 $g_{j-1}$ 表示模 $\delta$；它提供该列的前置坐标。所有坐标先模 $\delta$ 归约，再被前面的列归约。构造出的列满足同余，其行列式为 $\delta/g_{d-1}$，恰是同余核的指数，因此生成完整原像。

对多行，通过计算 $w=F_iH\bmod\delta$ 的单行原像 $T$，把当前 $H\mathbb Z^d$ 与每条同余求交。新格的基是 $HT$。它仍包含 $\delta\mathbb Z^d$，因此其三角对角整除 $\delta$。非对角乘法可以模 $\delta$ 执行，因为减去 $\delta e_i$ 属于前面各列的格。用已构造的列归约得到规范列 HNF。存储的元素至多 $\delta$，每个乘积低于 $\delta^2$，每次模加/减立即归约。不维护完整的幺模变换。

提升 $(-FH/\delta;H)$ 使用带准入上界 $d(\max|F|+2\delta+1)$ 的商/余累加。有理重构使用交错的收敛项系数符号与 Euclidean 行列式不变量，独立于余数乘积地界定系数乘积。LCM 与分子缩放先检查再乘。

当前重构域使用素数 2147483647 与 2147483629、至多 $2^{30}$ 的分母窗口、低于 $2^{31}$ 的公分母，以及上述累加上界。奇异的定主元 chart 或重构失败会被报告；不存在 bigint 或浮点回退。这是一个充分的动态准入域，不是对任意精度整数矩阵的支持。

## 公共 API 与持久化

```python
space = ClusterSpace(
    primitive_atoms, symprec=1e-5,
    cutoffs={2: 4.0, 3: 3.0}, max_body_orders={2: 2, 3: 3},
)
mapping = ClusterMap(space, supercell_atoms)
system = FitSystem(ForceDataset(mapping, structures))
model = system.solve()
```

`ClusterSpace` 直接接受 ASE 原胞原子。`ClusterMap` 在未提供矩阵时从其 ASE 原子推断 supercell matrix。`ClusterMap` 从 `mlfcs.mapping` 与包根导出。`PrimitiveCell`、`Supercell`、Taylor calculator 与旧的 `prepare()` API 不属于当前 API。

原生力常数文件使用 HDF5 格式版本 5；更旧的原生文件被拒绝。显式数组在不重构轨道的情况下保留存储的物理参数化与质量。只有 ForceConstants 提供 save/load；工作区、JIT 缓存与对象图不被序列化。模型与映射的兼容性由调用者负责；不计算、不比较模型身份哈希。

`ForceDesign.allocate_workspace()` 返回调用者持有的 scratch。流式拟合跨快照复用它。并发操作必须使用各自的工作区。把 Numba 线程数提高到工作区允许之上会抛错。Taylor calculator 共享编译代码并在运行时提供材料数组，避免按模型编译。张量阶是运行时参数；当前只有力设计轨道块使用 `prange`，各轨道块参数列不相交。

## 验证与测量

测试对照原 Python 候选标签/顺序、张量收缩与 SymPy 饱和格。冻结的对称性与力设计 fixture 覆盖多个阶、结构与剪切胞。精确残差、rank 与饱和检查守护精确代数路径。生产代码不 import SymPy 或 Rust；SymPy 位于 reference 依赖组，原生二进制不进入 wheel。

性能测量依赖具体负载。始终把冷 JIT 与热身执行分开报告，记录线程数与峰值内存，并比较完整的构造/design 路径以及单个 kernel。微基准不构成对所有材料的性能结论。
