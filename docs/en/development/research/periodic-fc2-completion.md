---
title: Exact-R and Periodic FC2 Completion
audience:
  - advanced
  - developer
status: experimental
code_verified: 4.0.0a6
---

# Exact-$R$ and Periodic FC2 Completion

This experimental option adds a source-supercell periodic harmonic Hessian next to canonical
exact-$R$ FC2. It defines

$$
E_C=\frac12\mathbf u^T\Phi_C\mathbf u,
\qquad
\mathbf F_C=-\Phi_C\mathbf u,
$$

before constructing design columns, so it is not an arbitrary force-residual correction.

The finite space satisfies Hessian symmetry, source-compatible space-group symmetry, and ASR. The
completion is the orthogonal complement of the exact FC2 image:

$$
\mathcal H_{\rm SC}^{\rm ASR}=\mathcal H_E\oplus\mathcal H_C.
$$

Enable it with `periodic_fc2_completion=True` in `ForceConstantFitter`. It is off by default. The
sidecar can be reordered only inside the same source translation quotient and cannot be exported to
a different-size supercell. The full derivation, symfc comparison, memory analysis, and benchmark
script are in `research/periodic_fc2_completion/`.
