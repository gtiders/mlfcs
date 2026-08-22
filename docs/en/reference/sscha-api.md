---
title: SSCHA API
audience:
  - developer
status: experimental
code_verified: 4.0.0a4
---

# SSCHA API

Document `SSCHA`, harmonic ensembles, iteration history, direct calculator execution, and effective FC2 output.

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
    log_level: int = 0,
)

run(
    calculator: Calculator,
    *,
    progress=None,
    calculate_free_energy: bool = True,
) -> SSCHAResult | TemperatureSeriesResult[SSCHAResult]
~~~

The public workflow uses a direct ASE calculator and returns effective harmonic force constants plus retained diagnostics. It does not expose the finite-difference sow/reap protocol.
