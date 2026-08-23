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

This study asks only whether the current transferable FC2 realization map

$$
M:\Theta_{\mathrm{primitive}}\rightarrow\mathcal H_{\mathrm{SC}}
$$

can satisfy

$$
\dim\ker M=0,
\qquad
\dim\operatorname{im}M<\dim\mathcal H_{\mathrm{SC}}.
$$

The result has two layers. KCl does reach this sweet spot at the representation level, but the
prescribed center-of-mass-free data do not identify the complete transferable-plus-closure space.
The overall outcome is therefore **No-Go**, with no production architecture work started.

## Finite observable space

The case uses the two-atom KCl primitive and a $2\times2\times2$, 16-atom reference. Finite FC2
is first represented by a translation-reduced compact Hessian,

$$
\Phi_{ab}([R])\in\mathbb R^{3\times3}.
$$

Primitive translation covariance is inherent in these coordinates. Averaging the compatible
finite space-group actions and imposing

$$
\Phi_{ab}([R])=\Phi_{ba}([-R])^T
$$

gives an orthogonal projector $P_{\rm SC}$ whose image defines $\mathcal H_{\rm SC}$. ASR,
Born–Huang, and Huang conditions are deliberately absent in this phase. Idempotence and symmetry
of the projector are checked to $10^{-9}$ so an atom-permutation or Cartesian convention error
cannot silently produce a plausible dimension.

## Transferable realization map

Each current `PrimitiveInteractionSpace` FC2 parameter is expanded into exact-$R$ tensors, folded
onto the reference, and projected into the orthonormal observable coordinates:

| Metric | Value |
|---|---:|
| Primitive transferable parameter dimension | 4 |
| $\operatorname{rank}(M)$ | 4 |
| $\dim\ker M$ | 0 |
| $\dim\mathcal H_{\rm SC}$ | 13 |
| Closure dimension | 9 |

The singular values are

$$
(4.89897949,\ 3.46410162,\ 1.73205081,\ 1.73205081),
$$

against a rank tolerance of $1.41\times10^{-14}$. The projection residual is
$2.01\times10^{-15}$. Thus the real KCl case strictly demonstrates an identifiable transferable
FC2 representation that does not span the complete finite Hessian space.

## Orthogonal closure

A single SVD supplies the left orthogonal complement $N$:

| Check | Value |
|---|---:|
| $\operatorname{rank}[M\ N]$ | 13/13 |
| $\|M^TN\|_2$ | $1.07\times10^{-32}$ |
| Minimum representation principal angle | $\pi/2$ |
| Random observable-coordinate relative error | $1.51\times10^{-16}$ |
| Random full-Hessian relative error | $1.57\times10^{-16}$ |

At the representation level every $\phi\in\mathcal H_{\rm SC}$ therefore has a unique
decomposition

$$
\phi=M\theta+N\eta.
$$

The closure means only finite-supercell harmonic response absent from the current transferable
FC2 basis. It is neither a unique infinite-lattice interaction nor a long-range FC2 model.

## Dataset identifiability

The actual data use the case PolyMLP, 100 Gaussian Cartesian snapshots with $0.01$ Å standard
deviation, random seed 42, and center-of-mass displacement removed from every frame. The design is

$$
X=\left[X_{\rm SC}M\quad X_{\rm SC}N\right].
$$

| Metric | Value |
|---|---:|
| Transferable dataset rank | 4/4 |
| Closure dataset rank | 9/9 |
| Joint dataset rank | 12/13 |
| Joint nullity | 1 |
| Condition number on the nonzero subspace | 6.26 |
| Minimum dataset principal angle | 0 |

The joint null vector has a design residual of $1.37\times10^{-16}$ but a maximum Hessian ASR
residual of $2.10$. It belongs to the uniform-translation sector that center-of-mass-free sampling
cannot observe, rather than to an error in the representation completion. Each block is full rank
alone, but the two blocks share one indistinguishable direction on this dataset.

The transferable-only force RMSE is $5.87\times10^{-3}$ eV/Å, while a minimum-norm joint fit lowers
it to $6.46\times10^{-4}$ eV/Å. A smaller residual does not remove the joint parameter gauge and
is not sufficient justification for a production feature.

## Aliasing negative control

The existing one-atom, $4.1$ Å cutoff, $1\times1\times1$ reference model has three primitive
parameters but realization rank one, giving $\dim\ker M=2$. The production
`validate_realization_identifiability()` correctly raises `InteractionAliasingError`. The prototype
does not disable that check or conceal primitive aliasing with closure parameters.

## Decision

The experiment establishes that the representation sweet spot is real and that its SVD complement
is numerically stable. It also establishes that the unconstrained full observable closure is not
jointly identifiable from the prescribed center-of-mass-free data.

Under the locked acceptance criteria, the decision is **No-Go**. No fitter, IFC schema, SCPH,
SSCHA, or export changes follow from this study. A future investigation would have to start as a
separate ASR-constrained observable-space study rather than integrating the present 13-dimensional
unconstrained result.

The complete prototype and machine-readable output are under
`research/fc2_observable_closure/`.
