---
title: Loop SCPH API
audience:
  - advanced
  - developer
status: experimental
code_verified: 4.0.0a6
---

# Loop SCPH API

The current implementation contains only the static quartic loop self-energy and returns a
temperature-dependent effective FC2. It does not include FC3 bubbles, a frequency-dependent
self-energy, or a transport solver.

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
    imaginary_modes: Literal["error", "absolute", "exclude"] = "absolute",
    asr_tolerance: float = 1e-6,
    symprec: float = 1e-5,
    time_reversal: bool = True,
    symmetry_tolerance: float | None = 1e-6,
)
```

| Parameter | Meaning |
|---|---|
| `fc2` | Initial harmonic IFC carrying order 2 and a valid `StructureRelation`. |
| `fc4` | IFC carrying order 4 whose primitive/reference frame matches FC2 exactly. It may be the same object as FC2. |
| `temperature` | One value in K, or a sequence; a sequence is checked for duplicates and run in ascending order. |
| `interpolation_multiplier` | Positive integer multiple of the reference quotient for the frequency criterion and the reported mesh. |
| `scph_multiplier` | Multiple of the loop covariance integration grid; it must be an integer multiple of `interpolation_multiplier`. |
| `statistics` | `quantum` uses Bose statistics with zero-point fluctuations, `classical` uses the classical limit. |
| `mixing` | In $(0,1]$; mixes the covariance of consecutive iterations. |
| `tolerance` | Stopping threshold for the frequency change in THz; the criterion is the star-weighted full-grid RMS described below. |
| `max_iterations` | Maximum iterations per temperature, at least 1. |
| `frequency_cutoff_thz` | Modes below this absolute frequency do not enter the covariance; must be non-negative. The three Gamma translations are not a cutoff question: they are removed from the modal space before any weight is formed. |
| `warm_start` | Optional initial effective FC2 that must be compatible with the input structure relation. |
| `continuation` | Whether a multi-temperature run initializes each temperature from the previous result. |
| `imaginary_modes` | What to do about a negative eigenvalue. `absolute` (the default) weights $\lvert\lambda\rvert$, because an unstable trial lattice is a normal SCPH starting point; `exclude` drops the mode, while `error` is an explicit diagnostic policy. There is no numerical “imaginary-frequency tolerance”. |
| `asr_tolerance` | Relative tolerance of the acoustic-sum-rule certificate $\lVert D(\Gamma)B\rVert$ against the scale of the grid. It is used to *name* the cause of a rejection, not to add a new one. |
| `symmetry_tolerance` | Relative tolerance of the full-star covariance and Hermiticity gates. `None` switches the comparisons off, but non-finite data is still refused. |
| `symprec` | Geometric tolerance used to identify the primitive symmetry. It is a different quantity from any dynamical-matrix tolerance and is recorded in the result. |
| `time_reversal` | Whether time reversal joins the star decomposition as an antiunitary member; turning it off can only add irreducible points. |

### Gamma, the cutoff and the covariance

The three acoustic translations are the null space of the mass-weighted translation operator, so
they are removed from the modal space *before* $1/\omega^2$ is evaluated, at every step and not
only when `frequency_cutoff_thz` happens to be non-zero. At $\Gamma$ the solver therefore
diagonalizes the dynamical matrix in the internal subspace and lifts the resulting covariance
back, which is why the covariance of a translation-invariant model is finite with or without a
cutoff.

The covariance gate is stricter than the dynamical-matrix gate: the weights go like
$1/\lambda$, so a matrix residual the matrix gate accepts can still leave the covariance
non-covariant by more than the same relative tolerance. The solver compares the expanded
covariance of every star member with the one built directly at that member's own q point, and
reports the member, the operation and the residual when they disagree.

The grid does not accept arbitrary triples. Its size is fixed by the reference supercell matrix
and the integer multiplier, so the q grid can never be incompatible with the periodic group of
the finite supercell.

## `run()`

```python
run() -> LoopSCPHResult | TemperatureSeriesResult[LoopSCPHResult]
```

A single temperature returns a `LoopSCPHResult`; several temperatures return an ascending
`TemperatureSeriesResult`. If the tolerance is not met within the iteration budget the solver
logs a warning and returns the last iterate with `converged=False`.

## Result objects

```python
from mlfcs.reciprocal.scph.solver import LoopSCPHIteration, LoopSCPHResult
```

`LoopSCPHIteration`:

- `index`: iteration number, starting at 1;
- `frequency_change_thz`: the stopping criterion, the star-weighted full-grid frequency RMS
  $\Delta\omega=\sqrt{(1/(N_qN_b))\sum_s w_s\lVert\omega_s^{(n)}-\omega_s^{(n-1)}\rVert_2^2}$;
- `correction_norm`: Frobenius norm of this iteration's loop FC2 correction.

`LoopSCPHResult`:

- `temperature`;
- `irreducible_qpoints`: the irreducible representative q points;
- `irreducible_frequencies`: frequencies at the representatives, in THz;
- `weights`: star weights, summing to the number of full grid points $N_q$;
- `grid`: the exact star decomposition of that grid (`IrreducibleReciprocalGrid`);
- `symprec`: the geometric tolerance used for the symmetry identification;
- `force_constants`: the effective FC2, ready for realization and export;
- `history`, `converged`;
- an `iterations` property equal to `len(history)`.

The result stores representatives only, and the full mesh must be expanded explicitly, so a
caller cannot mistake which grid a name refers to:

```python
result.full_qpoints()          # full-grid q points, in full-grid order
result.expand_frequencies()    # full-grid frequencies, per point equal to its representative
result.n_qpoints               # N_q
result.n_irreducible           # N_irr
result.reduction_ratio         # N_q / N_irr
```

`harmonic_frequencies` follows the same convention: it returns a `HarmonicMeshResult` with the
fields `irreducible_qpoints`, `irreducible_frequencies`, `weights`, `grid`, `symprec` and
`time_reversal`, and the same `full_qpoints()` and `expand_frequencies()` methods.

```python
result = LoopSCPH(fc2=source, fc4=source, temperature=600.0, mixing=0.3).run()
write_force_constants(result.force_constants, "T600K.h5", format="hdf5")
```

## Multi-temperature results

`TemperatureSeriesResult` supports iteration, integer indexing and `at_temperature(600)`;
`temperatures`, `results`, `iterations` and `converged` expose the schedule and the values.
Requesting a temperature that was not scheduled raises `KeyError`.
