---
title: Fitting API
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Fitting API

Signatures and contracts for `ForceConstantFitter`, `FitDataset`, fit results, batch size, cutoffs, and diagnostics.

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

One fitter accepts one fixed reference supercell. Every training structure must preserve its lattice, atom count, labels, and atom order.

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

`batch_size` controls streamed design construction and does not repeat the fit. The stable strict path uses `regularization=None`; unsupported names are rejected.
