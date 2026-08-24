---
title: 力拟合 API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 力拟合 API

## `ForceConstantFitter`

```python
ForceConstantFitter(
    primitive: Atoms,
    reference: Atoms,
    *,
    orders: tuple[int, ...] = (2, 3),
    cutoffs: dict[int, float | int | None] | None = None,
    max_body_orders: dict[int, int | None] | None = None,
    fitting_basis: Literal["taylor", "wick"] = "taylor",
    symprec: float = 1e-5,
    jax_platform: Literal["auto", "cpu", "gpu"] = "auto",
)
```

| 参数 | 含义 |
|---|---|
| `primitive` | primitive ASE `Atoms`。 |
| `reference` | 唯一训练超胞；所有快照必须与其完全同构且顺序一致。 |
| `orders` | 连续 IFC 阶数，例如 `(2,)`、`(2,3)` 或 `(2,3,4)`；不允许跳阶。 |
| `cutoffs` | 每个拟合阶必须有一项；值可为正 Å、负整数壳层或 `None`。 |
| `max_body_orders` | 可选的逐阶 body-order 上限；缺失阶等价于 `None`。 |
| `fitting_basis` | `"taylor"` 为默认；`"wick"` 使用训练 covariance 定义坐标，最终仍 lowering 为 Taylor IFC。 |
| `symprec` | 结构映射与空间群容差。 |
| `jax_platform` | `auto` 使用 JAX 默认设备；显式 `cpu`/`gpu` 在设备不存在时拒绝。 |

`cutoffs=None` 这个“字典整体为 None”并不表示每阶 cutoff 为 None，而是缺少必需配置并会被拒绝；必须写成
`cutoffs={2: None}`。

## `fit()`

```python
fit(
    structures: list[Atoms] | tuple[Atoms, ...],
    *,
    batch_size: int = 1,
    validation_split: float = 0.1,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    seed: int = 0,
    acoustic_sum_rule: bool = True,
    precondition: bool = True,
    allow_unconverged: bool = False,
    regularization: Literal["scaled_group_lasso"] | None = None,
    cache_directory: str | Path | None = None,
) -> FittingResult
```

| 参数 | 含义 |
|---|---|
| `structures` | 每帧为 reference 的位移结构，必须能通过 `atoms.get_forces()` 取得 `(N,3)` 有限力。 |
| `batch_size` | 1–4；仅控制 streamed design/Gram 吞吐和内存。 |
| `validation_split` | $[0,1)$；按 `seed` 随机划分，四舍五入得到验证帧数。 |
| `tolerance` | 线性求解迭代停止容差，必须为正。 |
| `max_iterations` | 最大求解步数，必须为正。 |
| `seed` | 训练/验证划分的 NumPy 随机种子。 |
| `acoustic_sum_rule` | 在物理 Taylor 参数的线性约束空间中严格施加逐阶 ASR。 |
| `precondition` | 是否使用精确列范数预条件。 |
| `allow_unconverged` | 为 `False` 时未收敛抛错；为 `True` 时 warning 并返回最后参数。 |
| `regularization` | 默认 `None` 为严格最小二乘；可选 `scaled_group_lasso`。 |
| `cache_directory` | streamed Gram 恢复缓存目录；不改变数学结果。 |

ASR 构造在物理 Taylor 参数层，随后映射到所选拟合基，因此 Taylor/Wick 共用同一约束语义。

## `FittingResult`

```python
from mlfcs.fitting import FittingResult
```

主要字段：

| 字段 | 含义 |
|---|---|
| `force_constants` | lowering 后的 canonical Taylor `ForceConstants`。 |
| `fitting_parameters` | 所选拟合基中的最终参数。 |
| `fitting_basis` | `"taylor"` 或 `"wick"`。 |
| `parameter_scale` | 求解使用的列尺度。 |
| `training_force_rmse`、`validation_force_rmse` | eV/Å。 |
| `training_relative_force_error`、`validation_relative_force_error` | 无量纲相对 L2 误差。 |
| `order_force_rms` | 每阶预测力贡献 RMS。 |
| `iterations`、`stop_code`、`residual_norm` | 求解器状态。 |
| `maximum_constraint_residual` | 最终联合约束最大残差。 |
| `maximum_reference_force` | reference 自带力的最大模，单位 eV/Å。 |
| `maximum_snapshot_net_force` | 快照净力最大模，单位 eV/Å。 |
| `maximum_center_of_mass_displacement` | 快照质心位移最大模，单位 Å。 |
| `lowered_fc1_maximum`、`lowered_fc1_net` | Wick lowering 产生的 FC1 诊断；Taylor 为 `None`。 |
| `lowering_force_maximum`、`lowering_force_relative` | Wick predictor 与 Taylor 输出的有限胞差异诊断。 |
| `design_kernel_signatures`、`design_tiles`、`static_device_bytes` | design 程序规模。 |
| `gram_feature_passes`、`prediction_feature_passes` | 特征执行次数。 |

结果对象不提供写文件方法；使用 `write_force_constants(result.force_constants, ...)`。
