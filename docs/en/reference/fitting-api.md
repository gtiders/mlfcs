---
title: Fitting API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# Fitting API

## `ForceConstantFitter`

```python
ForceConstantFitter(
    primitive: Atoms,
    reference: Atoms,
    *,
    orders: tuple[int, ...] = (2, 3),
    cutoffs: dict[int, float | int | None] | None = None,
    max_body_orders: dict[int, int | None] | None = None,
    fitting_basis: Literal["taylor", "wick"] = "taylor",
    periodic_fc2_completion: bool = False,
    periodic_fc2_rank_tolerance: float | None = None,
    symprec: float = 1e-5,
    jax_platform: Literal["auto", "cpu", "gpu"] = "auto",
)
```

| Parameter | Meaning |
|---|---|
| `primitive` | Primitive ASE `Atoms`. |
| `reference` | The single training supercell. Every snapshot must use this cell and atom order. |
| `orders` | Consecutive IFC orders such as `(2,)`, `(2,3)`, or `(2,3,4)`. |
| `cutoffs` | Required per-order cutoff: positive Å, negative shell number, or `None`. |
| `max_body_orders` | Optional per-order body-order limit. |
| `fitting_basis` | `"taylor"` by default; `"wick"` is lowered to Taylor IFCs after fitting. |
| `periodic_fc2_completion` | Add an optional source-periodic, symmetry- and ASR-allowed FC2 complement. |
| `periodic_fc2_rank_tolerance` | Optional absolute SVD rank threshold. `None` uses the reported FP64 criterion. |
| `symprec` | Structure-mapping and space-group tolerance. |
| `jax_platform` | `auto`, `cpu`, or `gpu`. |

The completion is a true finite-supercell Hessian, not a force-residual model. It requires FC2,
strict ASR, and unregularized least squares. It never suppresses `InteractionAliasingError` from the
exact-$R$ representation.

## `fit()`

```python
fit(
    structures,
    *,
    batch_size: int = 1,
    validation_split: float = 0.1,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    seed: int = 0,
    acoustic_sum_rule: bool = True,
    precondition: bool = True,
    allow_unconverged: bool = False,
    regularization: Literal["scaled_group_lasso"] | None = None,
    cache_directory: str | Path | None = None,
) -> FittingResult
```

`batch_size` controls streamed design/Gram throughput, not repeated fitting. `tolerance` is the
iterative solver threshold. `acoustic_sum_rule=True` constructs an exact parameter null space.

`FittingResult.force_constants` always contains Taylor IFCs. With completion enabled,
`force_constants.sparse[2]` remains the transferable exact-$R$ model and
`force_constants.periodic_fc2_completion` is the source-bound Hessian sidecar. `materialize(2)`,
phonopy output, and `MLFCSCalculator` use their sum. The sidecar cannot be realized onto a different
translation quotient because it has no unique exact-$R$ lift.
