---
title: 拟合 API
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# 拟合 API

记录 `ForceConstantFitter`、`FitDataset`、拟合结果、batch size、cutoff 和诊断的签名与契约。

## `ForceConstantFitter`

~~~python
ForceConstantFitter(
    primitive: Atoms,
    reference: Atoms,
    *,
    orders: tuple[int, ...] = (2, 3),
    cutoffs: dict[int, float | int | None] | None = None,
    max_body_orders: dict[int, int | None] | None = None,
    symprec: float = 1e-5,
    jax_platform: Literal["auto", "cpu", "gpu"] = "auto",
    verbose: bool = True,
)
~~~

一个 fitter 只接受一个固定 reference supercell。所有训练结构必须保持其晶格、原子数、标签和原子顺序。

~~~python
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
    regularization: str | None = None,
    cache_directory: str | Path | None = None,
) -> FittingResult
~~~

`batch_size` 控制流式 design 构造，不会重复拟合。稳定严格路径使用 `regularization=None`；不支持的名称会被拒绝。
