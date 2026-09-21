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
    symprec: float = 1e-5,
    time_reversal: bool = True,
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
| `tolerance` | 相邻两次频率变化的停止阈值，单位 THz；判据是星权重下的全网格 RMS（见下）。 |
| `max_iterations` | 每温度最多迭代数，至少 1。 |
| `frequency_cutoff_thz` | 低于该绝对频率的模态不进入协方差，必须非负。 |
| `warm_start` | 可选初始有效 FC2，必须与输入结构关系兼容。 |
| `continuation` | 多温度时是否用前一温度结果初始化下一温度。 |
| `symprec` | 识别原胞对称性的几何容差；与任何动力学矩阵数值容差是两回事，会被记录在结果里。 |
| `time_reversal` | 是否把时间反演作为反幺正成员参与星分解；关闭后不可约点只会更多。 |

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
from mlfcs.reciprocal.scph.solver import LoopSCPHIteration, LoopSCPHResult
```

`LoopSCPHIteration`：

- `index`：从 1 开始的迭代号；
- `frequency_change_thz`：停止判据，即星权重下的全网格频率 RMS
  $\Delta\omega=\sqrt{(1/(N_qN_b))\sum_s w_s\lVert\omega_s^{(n)}-\omega_s^{(n-1)}\rVert_2^2}$；
- `correction_norm`：本轮 loop FC2 修正的 Frobenius 合成范数。

`LoopSCPHResult`：

- `temperature`；
- `irreducible_qpoints`：不可约代表 q 点；
- `irreducible_frequencies`：代表点上的频率，THz；
- `weights`：星权重，求和等于完整网格点数 $N_q$；
- `grid`：该网格的精确星分解（`IrreducibleReciprocalGrid`）；
- `symprec`：识别对称性时使用的几何容差；
- `force_constants`：可直接 realization/写出的有效 FC2；
- `history`、`converged`；
- `iterations` 属性等于 `len(history)`。

结果只保存不可约代表点，完整网格必须显式展开，避免调用方误判 `qpoints` 指哪一套网格：

```python
result.full_qpoints()          # 完整网格 q 点，按完整网格顺序
result.expand_frequencies()    # 完整网格频率，逐点等于代表点频谱，不再对角化
result.n_qpoints               # N_q
result.n_irreducible           # N_irr
result.reduction_ratio         # N_q / N_irr
```

同一套约定也适用于 `harmonic_frequencies`：它返回 `HarmonicMeshResult`，字段为
`irreducible_qpoints`、`irreducible_frequencies`、`weights`、`grid`、`symprec`、`time_reversal`，
并同样提供 `full_qpoints()` 与 `expand_frequencies()`。

```python
result = LoopSCPH(fc2=source, fc4=source, temperature=600.0, mixing=0.3).run()
write_force_constants(result.force_constants, "T600K.h5", format="hdf5")
```

## 多温度结果

`TemperatureSeriesResult` 支持迭代、整数索引和 `at_temperature(600)`；`temperatures`、`results`、
`iterations`、`converged` 分别给出调度和值。请求未调度温度会抛出 `KeyError`。
