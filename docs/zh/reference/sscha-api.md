---
title: 结构采样与 SSCHA API
audience:
  - user
  - advanced
  - developer
status: experimental
code_verified: 4.0.0a6
---

# 结构采样与 SSCHA API

## `perturb_structures`

```python
perturb_structures(
    reference: Atoms,
    *,
    snapshots: int,
    method: Literal["gaussian", "harmonic"] = "gaussian",
    displacement: float = 0.01,
    force_constants: ForceConstants | None = None,
    temperature: float | None = None,
    statistics: Literal["quantum", "classical"] = "quantum",
    cutoff_frequency: float = 0.01,
    imaginary_modes: Literal["error", "absolute", "exclude"] = "error",
    imaginary_tolerance: float = 1e-6,
    max_displacement: float | None = None,
    random_seed: int | None = None,
) -> list[Atoms]
```

第一参数始终是要被扰动的 `reference`，不是 primitive。每帧保留 reference 原子顺序并写入
`mlfcs_configuration_id` 与 `mlfcs_sampling_method`。

### Gaussian 模式

`method="gaussian"` 时，每个 Cartesian 分量独立取标准差为 `displacement` Å 的正态位移，然后逐帧移除
原子位移算术平均。只允许使用 `snapshots`、`displacement` 与 `random_seed`；提供 FC2、温度、裁剪或非默认
谐波参数会被拒绝。

```python
snapshots = perturb_structures(
    reference, snapshots=100, method="gaussian", displacement=0.01, random_seed=42
)
```

### Harmonic 模式

`method="harmonic"` 要求 `force_constants` 含 FC2 与 relation，并要求 `temperature`。FC2 会先 realization 到
给定 reference，再按相容 q 点、质量和模态统计采样。

| 参数 | 含义 |
|---|---|
| `statistics` | quantum 或 classical。 |
| `cutoff_frequency` | 小于该正频率的平移/近零模不采样，THz。 |
| `imaginary_modes` | `error` 拒绝；`absolute` 用绝对频率；`exclude` 不采样虚频模。 |
| `imaginary_tolerance` | 判为虚频的容差，THz。 |
| `max_displacement` | 可选逐原子位移模裁剪上限，Å；裁剪会 warning。 |
| `random_seed` | 固定后保证同一 q 点顺序与参数下可复现。 |

harmonic 模式中 `displacement` 不控制宽度，必须保持默认 0.01，否则抛错以防误解。

## `SSCHA`

```python
SSCHA(
    atoms: Atoms,
    *,
    reference: Atoms,
    cutoff: float | None,
    temperature: float | Sequence[float] = 300.0,
    statistics: Literal["quantum", "classical"] = "quantum",
    snapshots: int | Literal["auto"] = 1000,
    max_iterations: int = 10,
    initial_displacement: float = 0.01,
    random_seed: int | None = None,
    symprec: float = 1e-5,
    cutoff_frequency: float = 0.01,
    imaginary_modes: Literal["error", "absolute", "exclude"] = "error",
    imaginary_tolerance: float = 1e-6,
    max_displacement: float | None = None,
    initial_force_constants: ForceConstants | None = None,
    acoustic_sum_rule: bool = True,
    mixing: float = 1.0,
    continuation: bool = True,
)
```

| 参数 | 含义 |
|---|---|
| `atoms` | primitive；与公共采样函数的首参数语义不同。 |
| `reference` | 唯一固定采样超胞和原子顺序。 |
| `cutoff` | 内部 FC2 fitter cutoff；允许正 Å、负壳层或 `None`。 |
| `temperature` | 单温度或序列；序列升序运行。 |
| `snapshots` | 每轮快照数；`auto` 取不少于约四倍参数方程覆盖所需的帧数。 |
| `max_iterations` | canonical 更新上限；0 仍执行初始 Cartesian bootstrap。 |
| `initial_displacement` | 无初始 FC2 时 bootstrap Gaussian 标准差，Å。 |
| `initial_force_constants` | 可选 FC2；提供后首轮直接 canonical sampling。 |
| `mixing` | 新拟合 FC2 与当前 FC2 的线性混合系数，范围 $(0,1]$。 |
| `continuation` | 多温度时是否以上一温度有效 FC2 warm-start。 |

虚频、统计、频率 cutoff、裁剪和随机种子语义与 harmonic `perturb_structures` 相同。SSCHA 只返回有效
谐波 FC2，不把输入或中间的高阶 IFC 塞进结果。

## 增量接口

```python
sample() -> list[Atoms]

step(
    calculator: Calculator,
    *,
    progress: Callable[[int, int], None] | None = None,
    calculate_free_energy: bool = True,
) -> SSCHAIteration

run(
    calculator: Calculator,
    *,
    progress: Callable[[int, int], None] | None = None,
    calculate_free_energy: bool = True,
) -> SSCHAResult | TemperatureSeriesResult[SSCHAResult]
```

`sample()` 只生成当前轮结构；`step()` 直接用 ASE Calculator 计算力与可选势能并完成一轮拟合。多温度对象
不允许逐步调用，必须调用 `run()`。`force_constants` 属性在至少完成一次拟合后返回当前有效 FC2，
`supercell_atoms` 返回 reference 副本，`current_iteration` 等于已有 history 长度。

## `SSCHAIteration` 与 `SSCHAResult`

每轮保存：`index`、`sampling`、自由能及误差、实际/谐波势能、q 点与模态计数、最小频率、最大采样位移、
裁剪原子数、拟合相对力误差、混合后和原始 FC2 相对变化。Cartesian bootstrap 中不适用的频率字段为
`None`。

`SSCHAResult` 保存 `temperature`、最终 `force_constants` 与完整 `history`。多温度返回
`TemperatureSeriesResult`，其访问方式与 SCPH 相同。
