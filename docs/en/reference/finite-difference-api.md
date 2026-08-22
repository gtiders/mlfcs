---
title: Finite-difference API
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Finite-difference API

Manually maintained signatures and contracts for `ForceConstantCalculation`, stencils, sow/reap, direct execution, and extrapolation.

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

`atoms` is the explicit primitive and `reference` is the authoritative atom order for all displaced structures and returned forces. A positive cutoff is measured in Å, a negative integer selects a neighbor shell, and `None` uses the largest supported periodic-image-unambiguous radius with the documented boundary margin.

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

`sow()` and positional `reap()` share one deterministic configuration order. Shape mismatches, missing configuration identifiers, non-finite forces, and incompatible atom ordering are rejected.
