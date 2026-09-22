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
`ForceConstantFitter.fit(acoustic_sum_rule=True)` 共用同一个逐阶 `TranslationalASRProjector`。两条路径都先在
正交 Cartesian orbit 基中得到无约束物理参数 $\theta_0$，再做欧氏投影
$\operatorname*{argmin}_{A\theta=0}\lVert\theta-\theta_0\rVert_2$。

拟合的 Gram 不编码 ASR，所以同一个物理 `GramStatistics` 可分别用于开启或关闭 ASR 的拟合。结果同时报告
投影前后的训练误差、ASR 残差、参数修正范数和投影迭代次数。

## `TranslationalASRProjector`

```python
from mlfcs.constraints import TranslationalASRProjector
```

`TranslationalASRProjector.from_orbit_space(orbit_space)` 为单个 IFC 阶数构造 Cartesian ASR 方程；
`project(parameters, tolerance=...)` 返回含投影参数和完整诊断的 `ASRProjectionResult`。这里的 tolerance 是
相对残差停止准则，不参与判定哪些约束系数存在。

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
