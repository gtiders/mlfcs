---
title: 平移与旋转约束 API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 平移与旋转约束 API

## 拟合与有限差分中的 ASR

`FiniteDifferenceCalculation.reap/run(acoustic_sum_rule=True)` 和
`ForceConstantFitter.fit(acoustic_sum_rule=True)` 都在参数重建/求解阶段施加平移声学求和规则。拟合侧先在
physical Taylor 参数空间直接构造约束，因此 ASR 与拟合参数具有相同的物理语义。

## `enforce_rotational_sum_rules`

```python
enforce_rotational_sum_rules(
    force_constants: ForceConstants,
    *,
    born_huang: bool = False,
    huang: bool = False,
    strength: float = 1.0,
    tolerance: float = 1e-8,
) -> RotationalSumRuleResult
```

该函数是独立 FC2 后处理，只修改 order 2；FC3 及以上逐项复制。它先严格投影 ASR，再在 ASR null space 内
求 Born–Huang/Huang 最小范数修正，最后再次去除浮点 ASR 残差。

| 参数 | 含义 |
|---|---|
| `force_constants` | 必须具有 relation 和 lattice-labelled sparse FC2。 |
| `born_huang` | 是否施加 FC2 Born–Huang 旋转不变条件。 |
| `huang` | 是否施加 Huang 应力平衡条件。 |
| `strength` | $[0,1]$；1 为严格完整修正，0 只保留严格 ASR。 |
| `tolerance` | 以中位最近邻长度无量纲化后的谱秩阈值，必须为正。 |

至少选择一个 `born_huang` 或 `huang`。该函数不把约束重新塞回原拟合 null space，也不改变 cutoff。

## `RotationalSumRuleResult`

```python
from mlfcs.constraints.rotational import RotationalSumRuleResult
```

字段包括：`force_constants`、`strength`、`tolerance`、`length_scale`、`retained_rank`，以及修正前后的
`acoustic_*`、`born_huang_*`、`huang_*` 残差；`relative_fc2_correction` 和
`maximum_fc2_correction` 衡量实际 FC2 改变量。未启用的条件残差为 `None`。

```python
corrected = enforce_rotational_sum_rules(fc2, born_huang=True, huang=True)
write_force_constants(corrected.force_constants, "corrected.h5", format="hdf5")
```
