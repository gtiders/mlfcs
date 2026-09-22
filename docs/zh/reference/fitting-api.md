---
title: 力拟合 API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 力拟合 API

`ForceConstantFitter` 只使用 Taylor 坐标。Gram 构造是独立的显式步骤，因此统计量可以手动
保存和复用，不需要保留训练快照或编译后的设计方案（design plan）。

```python
fitter = ForceConstantFitter(
    primitive,
    reference,
    orders=(2, 3),
    cutoffs={2: 5.4, 3: 4.5},
    max_body_orders={2: 2, 3: 3},
    periodic_fc2_completion=False,
    symprec=1e-5,
)
gram = fitter.prepare_gram(structures)
gram.save("training-gram.npz")
result = fitter.fit(gram, acoustic_sum_rule=True)
```

`prepare_gram()` 接收一份用户管理的数据集并返回可移植的充分统计量。它逐构型处理，
并行度来自相互作用 orbit，因此不再暴露构型批次大小；编译后的设计核规模由拟合问题决定，
而不是由 batch 决定。`GramStatistics.load()` 可在任意主机恢复。`fit()` 只负责求解和重建
Taylor IFC，不再隐式划分验证集或计算测试集预测。Gram 始终在完整物理坐标中构造，不携带 ASR 策略，
所以同一对象可以分别用于开启与关闭 ASR 的拟合。


`FittingResult` 保存拟合后的力常数、Taylor 参数、Gram 统计量、由 Gram 二次型得到的投影前后训练误差、
求解状态和 ASR 投影诊断。原始最小二乘参数保存在 `unprojected_parameters`，
`fitting_parameters` 是可选的逐阶欧氏 ASR 投影。模型力统一由 `MLFCSCalculator` 计算。

保存的 Gram 统计量携带物理设计身份。若结构、阶数集合、cutoff 或 orbit 参数化不同，读取后合并或拟合都会
明确拒绝。

## 参数的含义

拟合参数是每个 orbit 的正交 Cartesian 基 $Q$ 的系数，因此参数向量只有一种坐标含义：代表张量是
$Q\theta$，每个 orbit 的参数个数等于该 orbit 的不变子空间维数。精确整数系数 $c = R^{-1}\theta$ 与
$\theta$ 描述同一个张量，只是记录在规约晶胞的 lattice 参考系中，用于来源追踪与跨表示比较。

有限差分计划观测的分量行、观测矩阵与条件数随 orbit 一起保存（`observation_rows`、
`observation_matrix`、`observation_condition`）。它们决定哪些分量必须被测量，并决定由观测值求解参数的
条件数；参数向量本身不是观测值。详见[对称性与轨道](../theory/symmetry-and-orbits.md)。
