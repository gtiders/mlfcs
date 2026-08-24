---
title: SSCHA API
audience:
  - advanced
status: experimental
code_verified: 4.0.0a5
---

# SSCHA and sampling API

`SSCHA` is the iterative workflow. Structure generation outside that workflow uses the same sampler through `perturb_structures`:

```python
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
```

Gaussian sampling removes each snapshot's center-of-mass displacement. Harmonic sampling requires FC2 and temperature, realizes FC2 in `reference`, and uses the same mode pairing, frequency cutoff, imaginary-mode policy, and clipping implementation as SSCHA.

Iteration statistics are direct fields of `SSCHAIteration`; there is no separate diagnostics object or public harmonic-ensemble class.
