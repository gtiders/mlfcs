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
保存和复用，不需要保留训练快照或 JAX operator。

```python
fitter = ForceConstantFitter(
    primitive,
    reference,
    orders=(2, 3),
    cutoffs={2: 5.4, 3: 4.5},
    max_body_orders={2: 2, 3: 3},
    periodic_fc2_completion=False,
    symprec=1e-5,
    jax_platform="auto",
)
gram = fitter.prepare_gram(structures, batch_size=1, acoustic_sum_rule=True)
gram.save("training-gram.npz")
result = fitter.fit(gram, acoustic_sum_rule=True)
```

`prepare_gram()` 接收一份用户管理的数据集并返回可移植的充分统计量；
`GramStatistics.load()` 可在 CPU 或 GPU 主机恢复。`fit()` 只负责求解和重建 Taylor IFC，
不再隐式划分验证集或计算测试集预测。

`FittingResult` 保存拟合后的力常数、Taylor 参数、Gram 统计量、由 Gram 二次型得到的训练误差、
求解状态、约束残差、正则化状态以及可选 periodic FC2 completion。模型力统一由
`MLFCSCalculator` 计算。

Periodic completion 要求 FC2、严格 ASR 和无正则最小二乘。可迁移 exact-$R$ FC2 保存在
`force_constants.sparse[2]`，source-owned 有限 Hessian 保存在
`force_constants.periodic_fc2_completion`。
