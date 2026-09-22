---
title: SSCHA API
audience:
  - advanced
status: experimental
code_verified: 4.0.0a6
---

# SSCHA and sampling API

`SSCHA` requires the caller's explicit reference supercell; it never constructs one from a matrix
or cutoff. Its `symprec` is passed unchanged to primitive symmetry identification and the
primitive/reference relation:

```python
solver = SSCHA(
    primitive,
    reference=reference,
    cutoff=4.0,
    snapshots=32,
    symprec=1e-5,
)
```

`SSCHA` is the iterative workflow. Harmonic structure generation outside that workflow uses the
reciprocal sampler explicitly:

```python
from mlfcs.reciprocal import perturb_structures

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
    max_displacement: float | None = None,
    random_seed: int | None = None,
) -> list[Atoms]
```

Independent Cartesian Gaussian perturbations are a leaf utility instead:

```python
from mlfcs.tools.gaussian import perturb_structures

snapshots = perturb_structures(reference, snapshots=100, displacement=0.01)
```

Gaussian sampling removes each snapshot's center-of-mass displacement. Harmonic sampling requires FC2 and temperature, realizes FC2 in `reference`, and uses the same mode pairing, frequency cutoff, imaginary-mode policy, and clipping implementation as SSCHA.

Iteration statistics are direct fields of `SSCHAIteration`; there is no separate diagnostics object or public harmonic-ensemble class. Each iteration records both the full grid size (`n_qpoints`) and the irreducible count (`n_irreducible`), because the sampler diagonalizes representatives only while every full q point keeps its own random degrees of freedom.

The sampler itself exposes `grid`, `irreducible_qpoints`, `irreducible_frequencies`, `weights` and `full_qpoints()`; `SamplingState` reports `n_qpoints` and `n_irreducible` next to the mode counts.
