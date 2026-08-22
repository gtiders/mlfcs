---
title: SCPH API
audience:
  - developer
status: experimental
code_verified: 4.0.0a4
---

# SCPH API

Document `LoopSCPH`, temperature-series behavior, iteration records, and effective FC2 results.

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
    verbose: bool = True,
    qpoint_workers: int = 1,
)

run() -> LoopSCPHResult | TemperatureSeriesResult[LoopSCPHResult]
~~~

A sequence of temperatures is sorted before execution. Continuation uses the previous effective FC2 as the next warm start, and each result contains an exportable effective harmonic `ForceConstants`.
