---
title: SSCHA API
audience:
  - developer
status: experimental
code_verified: 4.0.0a5
---

# SSCHA API

记录 `SSCHA`、统一结构采样、迭代历史、直接 calculator 执行和有效 FC2 输出。

~~~python
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

run(
    calculator: Calculator,
    *,
    progress=None,
    calculate_free_energy: bool = True,
) -> SSCHAResult | TemperatureSeriesResult[SSCHAResult]
~~~

公共工作流直接使用 ASE calculator，并返回有效谐波力常数和保留诊断。它不暴露有限差分的 sow/reap 协议。

## `perturb_structures`

~~~python
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
~~~

Gaussian 模式逐帧移除质心位移。harmonic 模式要求提供 FC2 和温度，并将 FC2 显式 realization 到 `reference`；它与 SSCHA 共用模态配对、频率 cutoff、虚频策略和位移裁剪实现。采样统计直接保存在 `SSCHAIteration`，不再存在独立 diagnostics 对象或公共谐波 ensemble 类。
