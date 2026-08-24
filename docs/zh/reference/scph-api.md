---
title: Loop SCPH API
audience:
  - advanced
  - developer
status: experimental
code_verified: 4.0.0a6
---

# Loop SCPH API

当前实现只包含静态四阶 loop 自能，输出温度相关有效 FC2；不包含 FC3 bubble、频率依赖自能或输运求解。

## `LoopSCPH`

```python
LoopSCPH(
    *,
    fc2: ForceConstants,
    fc4: ForceConstants,
    temperature: float | Sequence[float],
    interpolation_multiplier: int = 1,
    scph_multiplier: int = 2,
    statistics: Literal["quantum", "classical"] = "quantum",
    mixing: float = 0.1,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    frequency_cutoff_thz: float = 0.0,
    warm_start: ForceConstants | None = None,
    continuation: bool = True,
    qpoint_workers: int = 1,
)
```

| 参数 | 含义 |
|---|---|
| `fc2` | 包含 order 2 与有效 `StructureRelation` 的初始谐波 IFC。 |
| `fc4` | 包含 order 4、且 primitive/reference 与 FC2 完全一致的 IFC。可与 FC2 是同一对象。 |
| `temperature` | 单一 K 值或序列；序列会去重检查并升序运行。 |
| `interpolation_multiplier` | 相对 reference quotient 的频率判据/输出插值网格倍数，正整数。 |
| `scph_multiplier` | loop covariance 积分网格倍数，必须是 interpolation multiplier 的整数倍。 |
| `statistics` | `quantum` 使用 Bose 统计与零点涨落，`classical` 使用经典极限。 |
| `mixing` | $(0,1]$；混合相邻迭代 covariance。 |
| `tolerance` | 相邻两次全网格频率 RMS 变化阈值，单位 THz。 |
| `max_iterations` | 每温度最多迭代数，至少 1。 |
| `frequency_cutoff_thz` | 低于该绝对频率的模态不进入协方差，必须非负。 |
| `warm_start` | 可选初始有效 FC2，必须与输入结构关系兼容。 |
| `continuation` | 多温度时是否用前一温度结果初始化下一温度。 |
| `qpoint_workers` | q 点 CPU 线程数，至少 1；不改变 q 点集合和结果顺序。 |

网格不接受任意三元组。其尺寸由 reference supercell matrix 与整数 multiplier 确定，从而避免 q 网格与
有限超胞周期群不相容。

## `run()`

```python
run() -> LoopSCPHResult | TemperatureSeriesResult[LoopSCPHResult]
```

单温度返回 `LoopSCPHResult`；多温度返回升序 `TemperatureSeriesResult`。未在最大步数内满足 tolerance
时 warning 并返回最后迭代，`converged=False`。

## 结果对象

```python
from mlfcs.physics.scph.solver import LoopSCPHIteration, LoopSCPHResult
```

`LoopSCPHIteration`：

- `index`：从 1 开始的迭代号；
- `frequency_change_thz`：停止判据；
- `correction_norm`：本轮 loop FC2 修正的 Frobenius 合成范数。

`LoopSCPHResult`：

- `temperature`；
- `qpoints`：相容 reciprocal quotient 点；
- `frequencies`：最终网格频率，THz；
- `force_constants`：可直接 realization/写出的有效 FC2；
- `history`、`converged`；
- `iterations` 属性等于 `len(history)`。

```python
result = LoopSCPH(fc2=source, fc4=source, temperature=600.0, mixing=0.3).run()
write_force_constants(result.force_constants, "T600K.h5", format="hdf5")
```

## 多温度结果

`TemperatureSeriesResult` 支持迭代、整数索引和 `at_temperature(600)`；`temperatures`、`results`、
`iterations`、`converged` 分别给出调度和值。请求未调度温度会抛出 `KeyError`。
