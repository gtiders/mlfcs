# Q&A

## 有限差分重建能否直接传 NumPy 力数组？

不能。`FiniteDifference.reconstruct` 接收带有力结果的有序 ASE `Atoms` 序列。几何、原子顺序和力数据共同构成一条记录；单独的数组无法证明它对应哪个位移。

## 如何计算有限差分的力？

力计算在外部完成。将 `fd.displacements()` 按原顺序交给 ASE calculator 或外部程序，保存每帧的力，再构造 `ForceDataset(mapping, structures)`，最后调用 `fd.reconstruct(dataset)`。MLFCS 不提供 `evaluate()` 包装器，也不核对采样 metadata。

## 可以使用 MACE 或其他 ASE calculator 吗？

可以。MLFCS 不拥有 calculator。只要支持输入结构中的元素和条件，任意 ASE 兼容 calculator 都可在外部计算位移结构的力。拟合和有限差分均通过 `ForceDataset` 收集结果。

## 为什么有限差分输入必须保持顺序？

位移索引编码了测量的混合导数。外部计算时，应保持 `displacements()` 返回的次序。重建会检查帧几何与原子顺序并拒绝不匹配，不会推断新顺序。

## 如何继续外部有限差分计算？

受支持的工作流是：用相同的原胞模型、cluster space、显式参考超胞和映射重新生成确定性序列，然后按原顺序传入带力 ASE 帧。MLFCS 不定义序列化实验计划格式。在不同程序间传递计算时，使用合适格式保存 ASE 结构和力。

## 一个有限差分对象能计算多个阶次吗？

不能。一个 `FiniteDifference` 对象处理一个阶次。联合拟合多个阶次时，在一个 `ClusterSpace` 中为各阶指定截断和体阶限制，再构建一个 `FitSystem`。

## 拟合过程会计算力吗？

不会。`ForceDataset` 只读取 ASE `Atoms` 上已经保存的力，不调用附带的 calculator；`FitSystem` 只消费准备好的数据集。这样力的产生始终由用户控制，也适用于外部电子结构或机器学习势计算。

## 参考超胞起什么作用？

原胞 cluster space 定义模型中的相互作用和对称约化参数；显式参考超胞把这些原胞相互作用映射到训练原子列表。超胞尺寸与形状决定目标参数能否区分。在昂贵计算前用 `mapping.rank_info(...).require_full()` 检查。

## 为什么拟合要构造正规系统？

优化拟合路径在流式读取训练结构时累积充分统计量 `A.T @ A` 和 `A.T @ f`，避免保留可能非常大的设计矩阵，也便于合并或复用兼容系统。默认求解器是列缩放 MINRES，不是批量梯度下降的神经网络优化器。

## 如果有参数未被观测怎么办？

正规矩阵中精确为零的对角元表示该参数在训练设计中缺失。默认求解器会抛出具名错误。应增加能激发该方向的结构或位移，选择更有信息量的参考超胞，或调整模型。正则化不能创造数据中不存在的信息。

## ASR 和旋转条件属于拟合的一部分吗？

不属于。先拟合，再按需显式执行力常数后处理投影。这样线性力拟合与物理约束选择彼此分离。请检查投影报告及其对力常数的影响。

ASR 逐阶用浮点 LSMR 修复平移不变性；它不调用构造轨道不变基时的整数核。旋转投影只修改 FC2，通过实际原子间距构造 Born–Huang 一次矩及可选的 Huang 二次矩，再用 SVD 处理可分辨方向。Born–Huang 的齐次形式要求原子力平衡；Huang 还要求零应力，所以默认关闭。

旋转修正保留原有 ASR 残差，不会代替 ASR。需要两者时，先对模型执行 `enforce_asr()`，再对返回的 `force_constants` 执行 `enforce_rotation()`。`rtol` 控制 ASR 相对残差；`rank_rtol` 控制旋转奇异值截断，两者语义不同。详见[投影与参数度量](domain-v6.md#旋转与平衡条件)。

## 如何保存或导出结果？

使用 `ForceConstants.save(path)` 保存原生版本 5 的 HDF5 文件，用 `ForceConstants.load(path)` 读取。文件显式保存几何、质量、对称操作、轨道基和系数，其中晶格基 dataset 名为 `lattice_basis`，加载不重建 cluster space。旧原生文件被拒绝。互操作输出使用 `ForceConstants.write(path, mapping, format=..., order=...)`；用户必须提供对应物理参数布局的 mapping，程序不检查跨对象兼容性。

折叠秩由 `ClusterMap.rank_info()` 报告；它使用 mapping 内部的精确秩认证。饱和整数核用于 ClusterSpace 的 stabilizer 不变基构造。跨领域的整数矩阵原语位于 `mlfcs.foundation.integer`，倒格点标签变换由 `mlfcs.phonon.grid` 内部完成。`Orbit.lattice_basis` 表示 stabilizer 不变晶格基；这些定义写入 docstring，名称按数学对象和操作命名。
