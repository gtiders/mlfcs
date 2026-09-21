---
title: Finite Difference API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# Finite Difference API

## `FiniteDifferenceCalculation`

```python
FiniteDifferenceCalculation(
    atoms: Atoms,
    *,
    order: int,
    reference: Atoms,
    cutoff: float | int = -5,
    max_body_order: int | None = None,
    displacement: float = 0.01,
    symprec: float = 1e-5,
)
```

| Parameter | Meaning |
|---|---|
| `atoms` | Primitive ASE `Atoms`; the first positional argument is not the reference. |
| `order` | Target IFC order, at least 2. One object reconstructs one order. |
| `reference` | Explicit training/displacement supercell; it fixes the atom order and identifiability. |
| `cutoff` | Required: a positive distance in angstrom or a negative primitive neighbour-shell index. `None` is rejected; a reference only decides whether the model is identifiable, never how large the model is. |
| `max_body_order` | Largest number of distinct `(site, R)` labels allowed in a cluster; `None` means no extra restriction. |
| `displacement` | Base central-difference step in angstrom, 0.01 by default. |
| `symprec` | Structure-matching and space-group tolerance. |

Construction builds the interaction/orbit space; the symmetry-reduced displacement plan is created on the
first access to `plan`, `manifest`, or on the first call to `sow()`.

## Plan identity and fingerprint

Displaced structures and their forces can be separated by hours or days (an external DFT code, a NEP
potential, a batch queue), and by then the Python object that owned the plan is usually gone. A
configuration count plus a zero-based id cannot say which plan a force belongs to: two plans of the same
crystal, order, cutoff and displacement can require the **same number** of displacements while measuring
**different component rows**.

Every plan therefore describes itself with a `DisplacementManifest`:

| Field | Content |
|---|---|
| `schema_version` | Manifest document and layout version, hashed with everything else. |
| `primitive_fingerprint` / `reference_fingerprint` | Structure fingerprints of the primitive and the reference. |
| `order`, `cutoff_angstrom`, `max_body_order`, `symprec`, `displacement_angstrom` | Model and difference settings. |
| `derivative_backend`, `stencil_signs`, `extrapolation` | Difference backend, stencil signs and the zero-step extrapolation description. |
| `orbits` | Per canonical orbit: `representative`, `dimension`, `observation_rows`, integer `exact_lattice_basis` and `images`. |
| `displacement_keys` | The observed keys, sorted. |
| `configurations` | Per configuration: id, key, displaced atoms, directions, signs and step. |

`fingerprint` is a SHA-256 hash of that field set (excluding the fingerprint itself) serialized as one
canonical JSON document with `sort_keys=True` and `separators=(",", ":")`. Floats are encoded with
`float.hex()`, so platform, endianness and the float-to-decimal algorithm cannot change the value, and no
`repr()`, object hash, memory layout or raw float bytes enter it. Structure fingerprints use the same
canonical form over atom numbers, cell, wrapped scaled positions and periodicity; identity is exact, so a
structure whose coordinates differ in the last bit is a different plan.

`DisplacementManifest.save()` and `load()` round-trip the document; loading recomputes the fingerprint and
rejects a hand-edited file.

## `sow()`

```python
sow() -> DisplacementBatch
```

Returns a `DisplacementBatch`: iterable and sized, whose `structures` are the displaced ASE `Atoms`, each
carrying `mlfcs_configuration_id` (zero-based) and `mlfcs_plan_fingerprint` in `info`. A cross-machine or
cross-day workflow saves the manifest first:

```python
batch = calculation.sow()
batch.manifest.save("displacements.json")
for atoms in batch:
    write(f"disp-{atoms.info['mlfcs_configuration_id']:04d}.vasp", atoms, vasp5=True)
```

## `reap()`

```python
reap(
    forces: ForceBatch,
    *,
    acoustic_sum_rule: bool = True,
) -> ForceConstants
```

`reap()` accepts **only** a `ForceBatch`, whose fields are `fingerprint`, `configuration_ids`, `forces`
and `schema_version`. The ids may arrive in any order: the receiving calculation restores the plan order
before differentiating. `forces` must have shape `(n_configurations, n_reference_atoms, 3)` and be
finite.

All of the following fail before any differentiation, with an exception that names the concrete
mismatch:

- a bare `ndarray`, a positional `list`/`tuple`, or a `Mapping` keyed by numeric configuration id: none
  of them carries a plan fingerprint, and the message says to regenerate the displacements with `sow()`;
- a fingerprint or schema version that does not match the plan;
- missing, duplicated or unknown configuration ids;
- a wrong atom count or force shape.

There is no "assume positional order when the fingerprint is missing" fallback: a bare sequence produced
by an old `sow()` cannot be accepted silently, and the displacements have to be regenerated.

## External calculator workflow

Codes such as VASP or Quantum ESPRESSO return forces as files, so the plan is remembered through its
manifest and the forces are handed back as a `ForceBatch`:

```python
from mlfcs.finite_difference.plan_identity import DisplacementManifest, ForceBatch

manifest = DisplacementManifest.load("displacements.json")   # saved next to the displacements
ids, forces = [], []
for index in range(len(manifest.configurations)):
    ids.append(index)
    forces.append(read_forces_from_external_code(index))      # (n_reference_atoms, 3)
result = calculation.reap(
    ForceBatch(
        fingerprint=manifest.fingerprint,
        configuration_ids=tuple(ids),
        forces=np.asarray(forces),
    )
)
```

As long as `calculation` was built from the same primitive, reference, order, cutoff, body order,
symprec, displacement and derivative backend, the fingerprint is recomputed to the same value; changing
any of them changes the fingerprint and the old forces are refused. Handing an ASE `Calculator` to
`run()` performs the same binding automatically, without writing a `ForceBatch` by hand.

## `evaluate()` and `run()`

```python
evaluate(
    calculator: Calculator,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> ForceBatch

run(
    calculator: Calculator,
    *,
    progress: Callable[[int, int], None] | None = None,
    acoustic_sum_rule: bool = True,
    derivative_backend: Literal["central", "extrapolate"] = "central",
    extrapolation_spacing: float | None = None,
    extrapolation_side_steps: int = 1,
    extrapolation_degree: int = 1,
) -> ForceConstants
```

`calculator` must be an ASE `Calculator`. `evaluate()` computes the forces of the central plan only and
returns a `ForceBatch` bound to the plan; `run()` performs the calculation and reconstruction serially.
`progress(done, total)` is called after every force evaluation.

`derivative_backend="extrapolate"` runs the whole central plan at several positive steps and extrapolates
to zero step with a polynomial in $h^2$:

- `extrapolation_spacing`: spacing between neighbouring steps, required and positive;
- `extrapolation_side_steps`: layers on both sides of the base step, at least 1;
- `extrapolation_degree`: fit degree in $h^2$, at least 1;
- every generated step must stay positive and the sample count must support the chosen degree.

Supplying any non-default extrapolation argument with the central backend is rejected, so a parameter
cannot be ignored silently.

```python
calculation = FiniteDifferenceCalculation(
    primitive, order=2, reference=reference, cutoff=7.7237404951
)
fc2 = calculation.run(calculator, acoustic_sum_rule=True)
```

The returned `ForceConstants.metadata` records the order, the resolved cutoff, the displacement, the
space group, ASR, the configuration count, the derivative backend, and the plan `plan_fingerprint` and
`plan_schema_version`.

## How reconstruction recovers parameters

Each orbit's `observation_rows` are the component rows the plan has to observe, and reconstruction
solves $Q_{\mathrm{obs}}\theta = y_{\mathrm{obs}}$ explicitly, where $Q_{\mathrm{obs}}$ is that
orbit's observation matrix and $y_{\mathrm{obs}}$ are the finite-difference derivative components.
The solution $\theta$ consists of coefficients of $Q$ and is expanded by
`expand_primitive_parameters` into force constants with site and integer-translation labels. These rows
are not parameter pivots: an observed component generally differs from a parameter, and the difference
is controlled by the condition number of $Q_{\mathrm{obs}}$, stored per orbit as
`observation_condition`.

## Intentional breaks in this cycle

The branch is in development and offers no compatibility layer, so a migration has to account for:

- `cutoff=None` is gone: pass an explicit radius or a negative shell index;
- `sow()` returns a `DisplacementBatch` instead of a bare list;
- `evaluate()` returns a `ForceBatch` and `reap()` accepts only that object;
- an old `sow()` result carries no fingerprint and can no longer be reaped; regenerate the displacements;
- error messages point at the new API, and there is no compatibility switch or silent fallback.
