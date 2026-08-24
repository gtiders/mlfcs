---
title: 有限差分 API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 有限差分 API

## `FiniteDifferenceCalculation`

```python
FiniteDifferenceCalculation(
    atoms: Atoms,
    *,
    order: int,
    reference: Atoms,
    cutoff: float | None = -5,
    max_body_order: int | None = None,
    displacement: float = 0.01,
    symprec: float = 1e-5,
)
```

| 参数 | 含义 |
|---|---|
| `atoms` | primitive ASE `Atoms`。首个位置参数不是 reference。 |
| `order` | 目标 IFC 阶数，必须至少为 2。一次对象只重建一个阶。 |
| `reference` | 显式训练/位移超胞，决定原子顺序和可辨识性。 |
| `cutoff` | 正 Å、负整数壳层或 `None` 安全最大半径。 |
| `max_body_order` | cluster 中允许的最大不同 `(site,R)` 数；`None` 表示不额外限制。 |
| `displacement` | 中心差分基础步长，单位 Å，默认 0.01。 |
| `symprec` | 结构对应与空间群容差。 |

构造阶段建立 interaction/orbit space；首次访问 `plan` 或调用 `sow()` 时才生成对称约化位移计划。

## `sow()`

```python
sow() -> list[Atoms]
```

返回带位移的结构列表。每帧包含零起始 `mlfcs_configuration_id`。位置式 `reap()` 要求力严格保持该顺序；
映射式 `reap()` 可以按 configuration ID 无序提交。

## `reap()`

```python
reap(
    forces: np.ndarray | Sequence[np.ndarray] | Mapping[int, np.ndarray],
    *,
    acoustic_sum_rule: bool = True,
) -> ForceConstants
```

数组形状必须为 `(n_configurations, n_reference_atoms, 3)`；单帧 sequence 的每项必须为
`(n_reference_atoms, 3)`。所有值必须有限。`acoustic_sum_rule=True` 在重建的参数空间中施加平移约束。

## `evaluate()` 与 `run()`

```python
evaluate(
    calculator: Calculator,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray

run(
    calculator: Calculator,
    *,
    progress: Callable[[int, int], None] | None = None,
    acoustic_sum_rule: bool = True,
    derivative_backend: Literal["central", "extrapolate"] = "central",
    extrapolation_spacing: float | None = None,
    extrapolation_side_steps: int = 1,
    extrapolation_degree: int = 1,
) -> ForceConstants
```

`calculator` 必须是 ASE `Calculator`。`evaluate()` 只计算 central plan 的力；`run()` 串行完成计算与重建。
`progress(done,total)` 在每次力计算后调用。

`derivative_backend="extrapolate"` 会在多个正步长执行完整 central plan，并以 $h^2$ 多项式外推到零步长：

- `extrapolation_spacing`：相邻步长间隔，必须显式给出且为正；
- `extrapolation_side_steps`：基础步长两侧的层数，至少 1；
- `extrapolation_degree`：关于 $h^2$ 的拟合次数，至少 1；
- 所有生成步长必须保持为正，采样点数必须足以支持所选次数。

central 模式下提供任何非默认 extrapolation 参数会被拒绝，避免参数被静默忽略。

```python
calculation = FiniteDifferenceCalculation(
    primitive, order=2, reference=reference, cutoff=None
)
fc2 = calculation.run(calculator, acoustic_sum_rule=True)
```

返回的 `ForceConstants.metadata` 记录 order、实际解析后的 cutoff、位移、空间群、ASR、构型数和导数后端。
