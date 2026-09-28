# Q&A

## 有限差分重建能否接收 NumPy 力数组？

不能。`FiniteDifference.reconstruct(structures)` 接收带有已存储力的有序 ASE `Atoms` 序列。几何和原子顺序用于确认每组力对应哪个生成位移。外部计算时，将力附到对应结构的 `SinglePointCalculator`，并保持 `fd.displacements()` 的序列顺序。

## `fd.evaluate(calculator)` 会复用结构上已存储的力吗？

不会。它会对每个生成几何请求新的力。需要 ASE calculator 计算这些结构时使用它。若力来自外部 DFT 或其他程序，将每组结果附到对应 ASE 结构，再调用 `reconstruct`。

## 可以使用 MACE、NEP 或外部 DFT 程序吗？

只要支持输入体系，任意 ASE 兼容 calculator 都可传给 `fd.evaluate()` 或 `SSCHA.run()`。拟合时先计算力，再把带力的 ASE 结构交给 `FitSystem.from_atoms`。不提供 ASE calculator 的外部程序也能使用：写出生成结构，在外部计算力，再将读回的力附到对应 ASE `Atoms`。

## 为什么有限差分必须保持输入顺序？

生成序列中的每个位置都对应特定的簇、符号组合和位移长度。重建会逐项检查几何和原子顺序，不会推断新顺序。不要对返回结构排序、去重或单独重排。

## 一个有限差分对象能计算多个阶次吗？

不能。`FiniteDifference(mapping, order=..., disps=...)` 处理一个阶次；可以指定多个不同位移长度并外推至零位移。若要重建多个阶次，每阶分别创建一个有限差分对象。一个 `ClusterSpace` 和 `FitSystem` 则可以联合拟合多个阶次。

## 拟合会计算力吗？

不会。`FitSystem.from_atoms` 读取 ASE 结构上已经存储的力，不调用其 calculator。用户可以自行选择 DFT、ASE 机器学习势或其他力计算流程。

## 超胞映射检查什么？

`ClusterMap` 将 `ClusterSpace` 与明确给定的 `Supercell` 连接起来。`mapping.rank_info(order).require_full()` 检查该超胞在结构层面能否区分模型参数，建议在昂贵计算前运行。映射满秩并不保证某一训练集具有足够多样的结构。

## 为什么默认拟合要构造正规系统？

`FitSystem` 流式累积各结构的充分统计量 $A^T A$ 和 $A^T f$，从而在不保留全部设计行的情况下求解和合并拟合。默认求解器是列缩放 MINRES。若算法需要原始方程，`FitData.from_atoms(...).arrays()` 可提供设计矩阵和力向量，以供直接最小二乘求解；这种方式需要保留更多内存。详见[拟合](fitting.md)。

## 如果拟合参数未被观测怎么办？

拟合正规矩阵中精确为零的对角元表示相应设计列缺失。默认求解会抛出 `UnobservedParameterError`。应增加有信息量的结构、调整超胞或修改模型。正则化不能补充训练数据中不存在的信息。

## 如何选择或更改质量？

质量取自构造 `ClusterSpace` 时使用的 ASE `Atoms`。若需指定同位素或位点质量，应在创建簇空间前设置到该结构上，例如调用 `atoms.set_masses([...])`。质量随后随模型、映射和倒空间计算传递。目前 `Harmonic`、`SCPH` 和 `SSCHA` 没有独立的质量覆盖参数；声子计算还要求质量分布与倒空间星约化所使用的对称性相容。

## 拟合时会自动施加 ASR 和旋转不变性吗？

不会。它们是拟合或有限差分重建后对 `ForceConstants` 显式执行的投影。若要同时施加两者，先做 ASR，再做旋转投影；旋转步骤会保留模型已有的声学残差。参数和报告说明见[力常数指南](force-constants.md)。

## 如何处理偶极长程力？

显式使用 `Ewald`：对每个训练帧先减去其长程力，再拟合短程模型，最后将 `ewald.fc2` 加到拟合得到的 FC2 数组。Ewald 张量按构造满足 ASR。它的 FC2 导出不包含非解析 LO-TO 分裂；下游声子计算仍需单独提供 NAC 数据。详见[长程力](long-range-forces.md)和 [NaCl 示例](../../tutorial/NaCl/README.md)。

## SCPH 和 SSCHA 如何处理虚频？

SCPH 报告有符号虚频，并可在静态环图计算中使用负曲率绝对值构造协方差。SSCHA 采样需要正定的高斯试探 FC2；它会报告不稳定试探态，只有设置 `bootstrap_displacement` 后才会启用笛卡尔 bootstrap。因此两种方法对稳定性的要求不同。详见 [SCPH](scph.md) 和 [SSCHA](sscha.md)。

## 温度序列按什么顺序运行？

`SCPH.run_many()` 和 `SSCHA.run_many()` 都接收严格递增的温度序列，按高温到低温执行，并按温度升序返回结果。SCPH 将已收敛的 FC2 传给下一个低温。SSCHA 也会以前一结果热启动；若中间温度未收敛，会抛出 `SSCHAContinuationError` 并停止后续温度。

## 如何保存或导出力常数？

使用 `ForceConstants.save(path)` 和 `ForceConstants.load(path)` 保存、读取原生 `.mlfcs`；原生 pickle 文件只应从可信来源加载。模型格式导出使用 `ForceConstants.write(path, mapping, format=..., order=...)`。已有紧凑 FC2 数组可用 `write_phonopy(...)` 导出。支持的格式和阶次见[力常数](force-constants.md)。
