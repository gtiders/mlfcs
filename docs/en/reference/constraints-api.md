---
title: Constraint API
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Constraint API

Document translational constraint construction and the public FC2 rotational correction operation.

~~~python
enforce_rotational_sum_rules(
    force_constants: ForceConstants,
    *,
    born_huang: bool = False,
    huang: bool = False,
    strength: float = 1.0,
    tolerance: float = 1e-8,
) -> RotationalSumRuleResult
~~~

The operation corrects FC2 only and always retains translational ASR. `strength=1.0` is the strict default; values in $[0,1]$ scale the Born–Huang/Huang correction and diagnostics report the achieved residuals.
