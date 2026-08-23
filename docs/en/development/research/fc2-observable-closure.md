---
title: Minimal FC2 Finite-Supercell Observable Closure Study
audience:
  - developer
status: research
code_verified: 4.0.0a4
examples:
  - research/fc2_observable_closure
---

# Minimal FC2 Finite-Supercell Observable Closure Study

This study asks whether the transferable FC2 realization map

$$
M:\Theta_{\mathrm{primitive}}\rightarrow\mathcal H_{\mathrm{SC}}
$$

can be completed by a strictly orthogonal, source-only closure on the $2\times2\times2$ KCl
reference. Closure means only:

> finite-supercell harmonic response not represented by the current transferable FC2 basis

It is neither a long-range FC2 model nor a unique infinite-lattice interaction.

## Phase one: unconstrained observable space

Finite FC2 is represented by translation-reduced compact Hessian blocks

$$
\Phi_{ab}([R])\in\mathbb R^{3\times3},
$$

with compatible space-group and Hessian-permutation symmetry. The transferable dimension is four,
$\operatorname{rank}(M)=4$, $\dim\ker M=0$, and $\dim\mathcal H_{\mathrm{SC}}=13$, leaving a
nine-dimensional orthogonal closure. Thus the representation sweet spot is real.

For 100 center-of-mass-free Gaussian snapshots with $0.01$ Å standard deviation and random seed 42,
however, the joint design has rank 12/13. Its unique null direction has centered-design residual
$1.37\times10^{-16}$, ASR maximum 2.10, uniform-displacement force norm 14.56, and relative
projection into the ASR-allowed space of only $1.31\times10^{-15}$. It is therefore precisely the
ASR-forbidden uniform-translation response hidden by center-of-mass removal.

## Phase two: ASR inside the representation

ASR is imposed as a defining subspace, not as a post-hoc correction:

$$
\mathcal H_{\mathrm{SC}}^{\mathrm{ASR}}=\ker C_{\mathrm{ASR}}.
$$

Observable and production transferable null spaces are constructed independently, followed by

$$
M_{\mathrm{ASR}}=Z_{\mathrm{SC}}^T M Z_\theta.
$$

No one-dimensional reduction is assumed. The observable ASR constraint actually has rank two:

| Metric | Value |
|---|---:|
| Observable dimension | $13\rightarrow11$ |
| Transferable dimension | $4\rightarrow2$ |
| $\operatorname{rank}(M_{\mathrm{ASR}})$ | 2 |
| $\dim\ker M_{\mathrm{ASR}}$ | 0 |
| ASR-constrained closure dimension | 9 |
| $\operatorname{rank}[M_{\mathrm{ASR}}\ N_{\mathrm{ASR}}]$ | 11/11 |
| $\|M_{\mathrm{ASR}}^TN_{\mathrm{ASR}}\|_2$ | $1.02\times10^{-15}$ |

The production ASR basis and direct Hessian ASR null space differ by a maximum principal angle of
$4.48\times10^{-16}$. A random allowed Hessian is reconstructed with relative error
$4.39\times10^{-16}$; its ASR, permutation, and symmetry residuals remain below
$7.58\times10^{-15}$, $1.08\times10^{-15}$, and $8.01\times10^{-15}$.

Projecting the old closure first happens to give the same nine-dimensional subspace in this case,
with maximum principal angle $2.27\times10^{-15}$ to the rebuilt closure. Rebuilding directly in
the allowed space remains the correct definition because it does not inherit an unconstrained gauge.

## Dataset controls

| Displacement treatment | Representation | Design rank | Nullity |
|---|---|---:|---:|
| COM removed | Unconstrained | 12/13 | 1 |
| COM retained | Unconstrained | 13/13 | 0 |
| COM removed | ASR constrained | 11/11 | 0 |
| COM retained | ASR constrained | 11/11 | 0 |

The centered and uncentered ASR-constrained singular values agree term by term. Once ASR is built
into the model, a uniform displacement produces no force and COM removal no longer changes the
identifiable information. Transferable, closure, and joint ranks are 2/2, 9/9, and 11/11. The joint
condition number is 4.02 and the minimum block principal angle is 1.491 radians.

## Force reconstruction

| Model | Rank | RMSE (eV/Å) | ASR maximum | Unique |
|---|---:|---:|---:|---:|
| A. Transferable only | 4/4 | $5.87\times10^{-3}$ | $7.94\times10^{-1}$ | Yes |
| B. Transferable + unconstrained closure | 12/13 | $6.46\times10^{-4}$ | $9.76\times10^{-2}$ | No |
| C. ASR transferable + closure | 11/11 | $6.46\times10^{-4}$ | $1.27\times10^{-14}$ | Yes |

Model C preserves the reconstruction accuracy of B while removing its gauge and satisfying ASR.
The closure Hessian norm ratio is 0.303, but this is only a representation-residual ratio and must
not be interpreted as a long-range-force fraction.

## Decision and scope

Phase two is **GO**: the ASR-constrained closure is complete, stably separated from the transferable
span, and identifiable from the center-of-mass-free dataset in this real case. This only licenses a
future architecture discussion. No production fitter, IFC schema, SCPH, SSCHA, or export code was
changed.

The one-atom aliasing negative control remains three primitive parameters with realization rank one
and a two-dimensional kernel, correctly rejected by the production check. Closure must never hide a
kernel in the transferable representation.

The complete prototype and machine-readable results are in `research/fc2_observable_closure/`.

## Phase three: structure and architecture boundary

Phase three explains the $13\rightarrow11$, $4\rightarrow2$, and $9\rightarrow9$ dimensions.
The ASR restriction has rank two on both the transferable span and the old closure, giving

$$
\dim(T\cap\ker A)=2,
\qquad
\dim(C\cap\ker A)=7.
$$

Constraining the old blocks separately spans only nine dimensions, while the complete allowed space
has dimension eleven. The missing two dimensions mix ASR-violating transferable and closure vectors
whose violations cancel. The old and rebuilt closure share seven dimensions; the other principal
angles are 0.848 and 1.209 radians. The allowed space must therefore be built first and complemented
inside it.

The two independent constraints on the four transferable parameters are

$$
\theta_0+4\theta_1+2\theta_2=0,
$$

$$
4\theta_1+2\theta_2+\theta_3=0.
$$

Thus 4→2 constrains linear combinations across orbit parameters rather than deleting two physical
IFCs. The production pivoted-QR map, the prototype SVD map, and an explicit simple map span the same
null space, while exact-$R$ site and translation semantics remain unchanged.

The compact observable Euclidean metric expands to exactly eight times the full-Hessian Frobenius
metric. The symmetry/permutation and raw ASR projectors nevertheless do not commute: their
commutator 2-norm is 0.5. The implementation must construct their subspace intersection directly
rather than apply projections sequentially.

### Reference and cutoff behavior

At a fixed $4.4391$ Å cutoff:

| Reference | $\dim\mathcal H^{\rm ASR}_{\rm SC}$ | Transferable | Closure | Target closure ratio |
|---|---:|---:|---:|---:|
| $2^3$ | 11 | 2 | 9 | 0.3027 |
| $3^3$ | 22 | 2 | 20 | 0.2674 |

The closure dimension grows with the source quotient. There is no canonical injection from a small
reference closure into a larger one because each small-cell residue has multiple exact-$R$ lifts and
closure supplies no rule for selecting one. The current column-wise dense projector was not extended
to $4^3$: its 2304-dimensional raw compact space makes that research construction unsuitable as a
future production algorithm.

On the fixed $2^3$ reference, increasing the cutoff from 2.5 Å to 4.8 Å reduces closure dimension
from eleven to three and the target closure norm ratio from 1.0 to 0.0718. At 5.5 Å the constrained
transferable space develops a one-dimensional kernel and the production alias check rejects it.
Closure must not hide that kernel.

### Data and canonicality

All 36 combinations of three seeds, four frame counts, and three displacement scales have rank
11/11, with condition numbers from 3.95 to 5.51. Force noise of $10^{-4}$ eV/Å changes parameters
by only $3.82\times10^{-4}$ relatively, showing no exceptional amplification.

The SVD coefficient $\eta$ is not canonical. Any $N\rightarrow NQ$ accompanied by
$\eta\rightarrow Q^T\eta$ represents the same Hessian. A random basis rotation changes the closure
projector by only $1.21\times10^{-15}$ and the reconstructed Hessian by
$3.82\times10^{-16}$. Stable semantics belong to the source fingerprint, closure projector, and
reconstructed finite Hessian—not to backend-dependent coordinates.

### Architecture decision

Closure should not enter `OrderParameterization`, `SparseOrderForceConstants`, or canonical HDF5.
The recommended next prototype is a source-specific `FiniteHarmonicResponse` plus a lightweight
harmonic design block. Its columns would be solved jointly with existing orbit columns while reusing
the streamed Gram and solver. Transferable `ForceConstants` remain unchanged, and the source
residual cannot be realized or exported to a different reference.

The phase-three decision is

$$
\boxed{\text{Mathematical GO + Prototype Recommended}}.
$$

It is not yet Architectural GO, and Production GO is explicitly not granted. Detailed derivations
and data are in `research/fc2_observable_closure/phase3-report.md` and `results-phase3.json`.
