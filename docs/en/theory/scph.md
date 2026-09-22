---
title: Loop-SCPH
audience:
  - advanced
status: experimental
code_verified: 4.0.0a6
---

# Loop-SCPH

`LoopSCPH` applies the static quartic loop correction to FC2 and returns a temperature-dependent
effective FC2. FC2 and FC4 are separate `ForceConstants` objects and must describe the same
primitive/reference frame.

```python
result = LoopSCPH(
    fc2=fc2, fc4=fc4, temperature=600,
    interpolation_multiplier=1, scph_multiplier=2,
    mixing=0.1, tolerance=1e-10, max_iterations=100,
).run()
```

`result.force_constants` is a normal, FC2-only `ForceConstants` object.  It
can therefore be written through the ordinary export API without a
SCPH-specific conversion:

```python
write_force_constants(result.force_constants, "scph.h5", format="hdf5")
write_force_constants(result.force_constants, "FORCE_CONSTANTS_SCPH", format="phonopy")
write_force_constants(result.force_constants, "force_constants.xml", format="alamode")
```

The q grids are reciprocal quotients of integer multiples of the reference
supercell matrix. `interpolation_multiplier` controls the reported frequency
grid and `scph_multiplier` controls the loop integration grid; the latter must
be an integer multiple of the former.

The grid itself is still the exact quotient of the finite translation group, but **the
diagonalization happens only at irreducible representative points**. The operations that keep
the grid split it into stars, each star carrying one representative and its members, and a
member is related to its representative by the space-group representation $U_g(q)$:

$$
 D(gq)=U_g(q)\,D(q)\,U_g(q)^\dagger,\qquad
 W(gq_s)=U_g\,W(q_s)\,U_g^\dagger,
$$

where $W(q)=V\operatorname{diag}(\sigma^2)V^\dagger$ is the basis-independent covariance
matrix; a member reached through time reversal is complex conjugated first. At $\Gamma$ the
modal space is the *internal* subspace: the three mass-weighted translations are the null space
of the translation operator, so they are projected out before the eigenproblem and the
covariance is lifted back afterwards. They are therefore absent from $W(\Gamma)$ rather than
present with a weight of $1/\omega^2$, and a translation-invariant model has a finite
$\Gamma$ covariance with or without a frequency cutoff. The columns of an eigenbasis inside a
degenerate subspace are a gauge, so only projectors, frequencies and covariances are compared
across a star -- never columns. Frequencies and
every other spectral quantity are equal across a star, so the full mesh is *exactly* expanded
from the representative data, and the expansion never enters an eigensolver again. The
positional gauge turns the primitive reciprocal lattice translation between a member's stored
label and the *unreduced* image of its representative into the diagonal factor
$\Gamma_G=\operatorname{diag}(\exp[2\pi i\,G\cdot\tau_a])$, which the covariance expansion
has to apply as well; without it the matrix and the Fourier phase differ by a factor of order
one on exactly the members whose label reduction bites.

An SCPH iteration therefore works on representatives only:

1. build and diagonalize $D(q_s)$ at the irreducible representatives and form $C(T)$, whose
   covariance is summed over the full star expansion;
2. contract every FC4 entry with that covariance;
3. form the target $\Phi_2^{\text{target}}=\Phi_2^{\text{bare}}+\Delta\Phi_2$;
4. stop when the star-weighted full-grid RMS frequency change
   $\Delta\omega=\sqrt{\frac{1}{N_qN_b}\sum_s w_s\lVert\omega_s^{(n)}-\omega_s^{(n-1)}\rVert_2^2}$
   falls below the tolerance, which equals the RMS of the expanded change.

The covariance gate is stricter than the dynamical-matrix gate. The weights go like
$1/\lambda$, so a residual the matrix gate accepts at a relative tolerance is amplified by the
ratio of the largest to the smallest included eigenvalue, and the expanded covariance can
disagree with the one built directly at a member's own q point by more than that tolerance.
The solver checks the covariance itself, member by member, instead of inferring it from the
covariance of the matrix. A soft or imaginary eigenvalue is reported where it is -- q point,
mode index, eigenvalue, frequency, temperature -- and is never converted into a stable mode by
$\sqrt{\lvert\lambda\rvert}$ unless the caller asks for that policy explicitly.

The star weights $w_s$ are the star sizes and sum to $N_q$. The covariance sum is a *real*
sum over the full star: weights are used for statistics (free energy, mode counts, the
convergence norm) and never to scale a representative's matrix, because members differ by a
rotation and not by a scalar.

`mixing` is covariance under-relaxation. An iteration is accepted as converged when the RMS frequency
change on the interpolation grid is below `tolerance`. Imaginary frequencies are retained as a
physical diagnostic and do not add a separate stopping condition. This implementation is loop-only;
it does not include the frequency-dependent bubble self-energy.

## Performance

`scripts/benchmark_reciprocal_reduction.py` reproduces the comparison below. Diagonalizations
are counted as *matrices*, not as calls: one batched `eigh` over the representatives is
`N_irr` diagonalizations. The model is an isotropic spring network for FC2 plus an on-site
quartic term, with `interpolation_multiplier=2`:

| system | $N_q$ | $N_{\mathrm{irr}}$ | reduction | frequency diagonalizations | one SCPH sweep | sampler initialization |
|---|---|---|---|---|---|---|
| diamond 2x2x2 | 64 | 8 | 8.00 | 8 (64 on the full grid) | 9 (24 on the full grid) | 3 (8 on the full grid) |
| hcp 2x2x2 | 64 | 12 | 5.33 | 12 (64 on the full grid) | 12 (24 on the full grid) | 4 (8 on the full grid) |
| GaAs 3x2x2 | 96 | 34 | 2.82 | 34 (96 on the full grid) | 18 (36 on the full grid) | 6 (12 on the full grid) |

The free energy and the sampling itself never enter the eigensolver again: the reported
harmonic free-energy time is of order $10^{-4}$ s with zero diagonalizations, and one batch of
4096 snapshots takes $2\text{--}4\times10^{-2}$ s.

Before anything is expanded, every force-constant set that the irreducible path actually
expands passes the **full-star covariance gate**: the gate expands the representative matrices
with $W_m=\Gamma_m U_m\overline{W_s}^{[a_m]}U_m^\dagger\Gamma_m^\dagger$ and compares them member by
member with the dynamical matrices built directly at each member's own label. The little-group
condition only constrains the representative itself -- it fixes the representative by definition
-- so it is neither sufficient nor usable as the merge gate; it survives as a local diagnostic
for stabilizer problems. The tolerance is relative: the allowed residual is
$\texttt{symmetry\_tolerance}\cdot s$ with $s$ the largest infinity norm over both the direct and
the expanded stacks. `symmetry_tolerance=None` is the only way to switch the gate off; `NaN`,
`+inf` and negative values are rejected, because a comparison against them is always false and
would disable the gate by accident. `symprec` must be finite and strictly positive.

The gate builds the full grid of $D(q)$, but only $N_{\mathrm{irr}}$ matrices reach
`eigh/eigvalsh`; a benchmark has to report matrix builds, eigensolver matrices, star expansion and
the full Fourier sum, not only the diagonalization reduction.

The merge criterion is that the diagonalization count equals $N_{\mathrm{irr}}$ exactly, not
that the wall clock improves: a small cell still pays a one-off symmetry analysis, so it may
look slower than the full grid, and a low-symmetry cell may barely speed up at all. That is
the correct result and must not be manufactured by enlarging the symmetry group.

## Temperature continuation

For several temperatures, use temperature continuation. Temperatures are evaluated in ascending
order, and the effective FC2 from one temperature initializes the next one.

```python
results = LoopSCPH(
    fc2=fc2, fc4=fc4, temperature=[300, 600, 900],
    interpolation_multiplier=1, scph_multiplier=2,
    max_iterations=100,
).run()
```
