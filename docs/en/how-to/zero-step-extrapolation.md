---
title: Zero-step finite-difference extrapolation
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Zero-step finite-difference extrapolation

[中文] | English

The extrapolation backend reduces single-step truncation error while still computing only the
force-constant order selected by `ForceConstantCalculation`. It is available only for direct,
serial ASE Calculator execution:

```python
fc = calculation.run(
    calculator,
    derivative_backend="extrapolate",
    extrapolation_spacing=0.005,
    extrapolation_side_steps=2,
    extrapolation_degree=1,
)
```

`displacement` remains the central step `h0`. The backend constructs

```text
h(k) = h0 + k * extrapolation_spacing
k = -extrapolation_side_steps, ..., +extrapolation_side_steps
```

Every step must be strictly positive. Each step has a complete central-difference subplan, so the
calculator count is multiplied by `2 * extrapolation_side_steps + 1`.

Central differences have an even-power error expansion. The backend fits each contracted
derivative to

```text
D(h) = D0 + c2 h^2 + c4 h^4 + ...
```

and uses `D0`. `extrapolation_degree=1` is the default and usually the most robust choice. Degree
`p` retains terms through `h^(2p)` and requires more than `p` displacement steps. Higher degree is
not automatically more accurate when calculator forces contain noise.

Extrapolation is performed before symmetry reconstruction and before translational or rotational
sum-rule projection. MLFCS reports the maximum correction from the central step, relative L2
correction, polynomial fit residual, and final sum-rule drift. These diagnostics should be used to
detect a displacement grid that is too narrow, too large, or dominated by force noise.

The backend is intentionally absent from `sow()` and `reap()`: keeping the external workflow tied
to one deterministic displacement plan avoids ambiguous ordering and multiplication of external
calculation directories.
