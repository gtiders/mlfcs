---
title: 有限差分 API
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# 有限差分 API

人工维护 `ForceConstantCalculation`、stencil、sow/reap、直接执行和外推的签名与契约。

## `ForceConstantCalculation`

~~~python
ForceConstantCalculation(
    atoms: Atoms,
    *,
    order: int,
    reference: Atoms,
    cutoff: float | None = -5,
    max_body_order: int | None = None,
    displacement: float = 0.01,
    symprec: float = 1e-5,
    verbose: bool = True,
)
~~~

`atoms` 是显式 primitive，`reference` 的原子顺序是全部位移结构与返回力的权威标签。正 cutoff 的单位为 Å，负整数表示近邻壳层，`None` 使用带规定边界裕量的最大周期像无歧义半径。

~~~python
sow() -> list[Atoms]
reap(forces, *, acoustic_sum_rule: bool = True) -> ForceConstants
run(
    calculator: Calculator,
    *,
    progress=None,
    acoustic_sum_rule: bool = True,
    derivative_backend: Literal["central", "extrapolate"] = "central",
    extrapolation_spacing: float | None = None,
    extrapolation_side_steps: int = 1,
    extrapolation_degree: int = 1,
) -> ForceConstants
~~~

`sow()` 和位置式 `reap()` 共用一个确定性构型顺序。力形状错误、构型 ID 缺失、非有限力和不相容原子顺序均会被拒绝。
