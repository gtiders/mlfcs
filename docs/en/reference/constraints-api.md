---
title: Translational and rotational constraints API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# Translational and rotational constraints API

## ASR in fitting and finite differences

`FiniteDifferenceCalculation.reap/run(acoustic_sum_rule=True)` and
`ForceConstantFitter.fit(acoustic_sum_rule=True)` use the same order-local
`TranslationalASRProjector`. Both first recover unconstrained coefficients in the physical
orthonormal Cartesian orbit basis and then apply the Euclidean projection
$\operatorname*{argmin}_{A\theta=0}\lVert\theta-\theta_0\rVert_2$.

Fitting does not encode ASR in the Gram matrix. Consequently one physical
`GramStatistics` object can be reused for fits with ASR enabled or disabled. Results report the
unprojected and projected training errors, the ASR residual before and after projection, the
parameter correction norm, and the projection iteration count.

## `TranslationalASRProjector`

```python
from mlfcs.constraints import TranslationalASRProjector
```

`TranslationalASRProjector.from_orbit_space(orbit_space)` builds the Cartesian ASR equations for one
IFC order. `project(parameters, tolerance=...)` returns an `ASRProjectionResult` containing the
projected parameters and all projection diagnostics. The tolerance is a relative residual stopping
criterion, not a rule for deciding which coefficients exist.

## `enforce_rotational_sum_rules`

```python
enforce_rotational_sum_rules(
    force_constants: ForceConstants,
    *,
    born_huang: bool = False,
    huang: bool = False,
    strength: float = 1.0,
    tolerance: float = 1e-8,
) -> RotationalSumRuleResult
```

This is a separate FC2 postprocessor. It changes order 2 only and copies FC3 and higher orders. It
first projects onto ASR, solves the minimum-norm Born-Huang/Huang correction within the ASR null
space, and removes the final floating-point ASR residual.

At least one of `born_huang` and `huang` must be selected. `strength` lies in $[0,1]$; one applies
the full correction and zero retains only strict ASR. `tolerance` is the spectral-rank threshold
after lengths have been normalized by the median nearest-neighbour distance.

`RotationalSumRuleResult` contains the corrected force constants, rank and length-scale information,
the residuals before and after correction, and the absolute and relative FC2 corrections.
