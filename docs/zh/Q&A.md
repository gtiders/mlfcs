# Q&A

## 有限差分重建能否直接传 NumPy 力数组？

不能。`FiniteDifference.reconstruct` 接收带有力结果的有序 ASE `Atoms` 序列。几何、原子顺序和力数据共同构成一条记录；单独的数组无法证明它对应哪个位移。

## `fd.evaluate(calculator)` 会使用缓存的力吗？

不会。它会针对每个生成结构明确请求一次新的力计算，并把结果保存在返回结构上。若 calculator 应计算当前几何，请使用此接口。若力来自外部程序，则将它们附为 ASE 已存储结果后直接调用 `reconstruct`。

## 可以使用 MACE 或其他 ASE calculator 吗？

可以。MLFCS 不拥有 calculator。只要支持输入结构中的元素和条件，任意 ASE 兼容 calculator 都可用于 `fd.evaluate(calculator)`。拟合时则先用所选 calculator 计算力，再将 ASE 结构交给 `FitSystem.from_atoms`。

## 为什么有限差分输入必须保持顺序？

位移索引编码了测量的混合导数。外部计算时，应保持 `displacements()` 返回的次序。重建会检查帧几何与原子顺序并拒绝不匹配，不会推断新顺序。

## 可以 pickle 有限差分对象并重新加载吗？

受支持的工作流是：用相同的原胞模型、cluster space、显式参考超胞和映射重新生成确定性序列，然后按原顺序传入带力 ASE 帧。MLFCS 不定义序列化实验计划格式。在不同程序间传递计算时，使用合适格式保存 ASE 结构和力。

## 一个有限差分对象能计算多个阶次吗？

不能。一个 `FiniteDifference` 对象处理一个阶次。联合拟合多个阶次时，在一个 `ClusterSpace` 中为各阶指定截断和体阶限制，再构建一个 `FitSystem`。

## 拟合过程会计算力吗？

不会。`FitSystem.from_atoms` 只读取 ASE `Atoms` 上已经保存的力，不调用附带的 calculator。这样力的产生始终由用户控制，也适用于外部电子结构或机器学习势计算。

## 参考超胞起什么作用？

原胞 cluster space 定义模型中的相互作用和对称约化参数；显式参考超胞把这些原胞相互作用映射到训练原子列表。超胞尺寸与形状决定目标参数能否区分。在昂贵计算前用 `mapping.rank_info(...).require_full()` 检查。

## 为什么拟合要构造正规系统？

优化拟合路径在流式读取训练结构时累积充分统计量 `A.T @ A` 和 `A.T @ f`，避免保留可能非常大的设计矩阵，也便于合并或复用兼容系统。默认求解器是列缩放 MINRES，不是批量梯度下降的神经网络优化器。

## 如果有参数未被观测怎么办？

正规矩阵中精确为零的对角元表示该参数在训练设计中缺失。默认求解器会抛出具名错误。应增加能激发该方向的结构或位移，选择更有信息量的参考超胞，或调整模型。正则化不能创造数据中不存在的信息。

## ASR 和旋转条件属于拟合的一部分吗？

不属于。先拟合，再按需显式执行力常数后处理投影。这样线性力拟合与物理约束选择彼此分离。请检查投影报告及其对力常数的影响。

## 如何保存或导出结果？

使用 `ForceConstants.save(path)` 保存原生可信 pickle 格式，并用 `ForceConstants.load(path)` 读取。互操作输出使用 `ForceConstants.write(path, mapping, format=..., order=...)`。原生 pickle 文件必须来自可信来源。
