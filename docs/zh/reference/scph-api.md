---
title: SCPH API
audience:
  - developer
status: experimental
code_verified: 4.0.0a5
---

# SCPH API

记录 `LoopSCPH`、温度序列行为、迭代记录和有效 FC2 结果。

~~~python
LoopSCPH(
    *,
    fc2: ForceConstants,
    fc4: ForceConstants,
    temperature: float | Sequence[float],
    interpolation_multiplier: int = 1,
    scph_multiplier: int = 2,
    statistics: str = "quantum",
    mixing: float = 0.1,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    frequency_cutoff_thz: float = 0.0,
    warm_start: ForceConstants | None = None,
    continuation: bool = True,
    qpoint_workers: int = 1,
)

run() -> LoopSCPHResult | TemperatureSeriesResult[LoopSCPHResult]
~~~

温度序列会在运行前排序。continuation 使用前一温度的有效 FC2 作为下一温度的 warm start；每个结果都包含可导出的有效谐波 `ForceConstants`。
