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
gram = fitter.prepare_gram(structures, acoustic_sum_rule=True)
gram.save("training-gram.npz")
result = fitter.fit(gram, acoustic_sum_rule=True)
```

`prepare_gram()` 接收一份用户管理的数据集并返回可移植的充分统计量。它逐构型处理，
并行度来自相互作用 orbit，因此不再暴露构型批次大小；编译后的设计核规模由拟合问题决定，
而不是由 batch 决定。`GramStatistics.load()` 可在任意主机恢复。`fit()` 只负责求解和重建
Taylor IFC，不再隐式划分验证集或计算测试集预测。


`FittingResult` 保存拟合后的力常数、Taylor 参数、Gram 统计量、由 Gram 二次型得到的训练误差、
求解状态、约束残差以及可选 periodic FC2 completion。模型力统一由
`MLFCSCalculator` 计算。

Periodic completion 要求 FC2、严格 ASR 和无正则最小二乘。可迁移 exact-$R$ FC2 保存在
`force_constants.sparse[2]`，source-owned 有限 Hessian 保存在
`force_constants.periodic_fc2_completion`。

## 参数的含义

拟合参数是每个 orbit 的正交 Cartesian 基 $Q$ 的系数，因此参数向量只有一种坐标含义：代表张量是
$Q\theta$，每个 orbit 的参数个数等于该 orbit 的不变子空间维数。精确整数系数 $c = R^{-1}\theta$ 与
$\theta$ 描述同一个张量，只是记录在规约晶胞的 lattice 参考系中，用于来源追踪与跨表示比较。

有限差分计划观测的分量行、观测矩阵与条件数随 orbit 一起保存（`observation_rows`、
`observation_matrix`、`observation_condition`）。它们决定哪些分量必须被测量，并决定由观测值求解参数的
条件数；参数向量本身不是观测值。详见[对称性与轨道](../theory/symmetry-and-orbits.md)。
