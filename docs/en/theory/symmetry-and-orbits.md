---
title: Symmetry and orbits
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# Symmetry and orbits

## Motivation

The space-group action on primitive sites and integer translations is affine:
$g: (\mathbf s, \mathbf t) \mapsto (R\mathbf s + \boldsymbol\tau,\ \mathbf t R^{\mathsf T} +
\boldsymbol\delta)$. Whether it can be decided *exactly* depends on the frame the rotation is written in.

* In the **lattice (scaled) frame** the spglib rotation matrix $R$ is an **integer** matrix for every
  cell.
* In the Cartesian frame $R_{\mathrm{cart}} = A^{-\mathsf T} R A^{\mathsf T}$ is integral only when the
  cell is axis aligned (orthorhombic, tetragonal, cubic). An fcc primitive $60^\circ$ cell and hexagonal
  or rhombohedral cells carry irrational entries such as $\sqrt3/2$.

Which tensor components symmetry admits must therefore be decided on integers, while what a
finite-difference plan observes and what a fit delivers is Cartesian physics. The two cannot share one
ambiguous `basis`.

## Mathematical object

Let $A_s$ be the user's primitive cell (rows are lattice vectors) and $A$ the reduced algebra cell. They
are related by one integer matrix $U$:

$$
A = U A_s, \qquad \lvert\det U\rvert = 1, \qquad U, U^{-1} \in \mathbb Z^{3\times 3}.
$$

One physical point, one integer translation and one rotation read

$$
\mathbf f_A = \mathbf f_s U^{-1}, \qquad \mathbf t_A = \mathbf t_s U^{-1}, \qquad
R_A = U^{-\mathsf T} R_s U^{\mathsf T}.
$$

The order-$n$ tensor map from lattice to Cartesian components has a single entry point:

$$
K_n = (A^{\mathsf T})^{\otimes n}.
$$

Every orbit carries a stabilizer set $S_a$ of its anchored cluster. With the 0/1 label-symmetric basis
$L$ (columns merge components equivalent under exchanges of equal atoms) the invariant subspace is

$$
V_{\mathbb Z} = \{\, x : (S_a L - L)x = 0 \ \ \forall a \,\} \subset \mathbb Z^{3^n},
$$

spanned by a **saturated integer basis** $B_{\mathbb Z}$. The physical tensor subspace follows from the
same frame change, $C = K_n B_{\mathbb Z}$, and a QR factorization with a positive diagonal

$$
C = QR, \qquad Q^{\mathsf T}Q = I .
$$

An orbit therefore owns two bases with disjoint roles:

| Field | Meaning | Used by |
|---|---|---|
| `exact_lattice_basis` | the integer basis $B_{\mathbb Z}$ | group algebra, rank certificate, invariant dimension, provenance |
| `cartesian_basis` | the orthonormal basis $Q$ | fitting, finite-difference reconstruction, ASR and other constraints, expansion |
| `coefficient_transform` | the upper triangular $R$ | conversion between exact and numerical coefficients, $c = R^{-1}\theta$ |

A fitted parameter $\theta$ has exactly one coordinate meaning: it is a coefficient of $Q$, and the
representative tensor is $Q\theta$. The exact integer coefficients $c = R^{-1}\theta$ are the same
physical tensor recorded in the lattice frame.

A finite-difference plan has to know which components it measures. Those rows are `observation_rows`,
chosen from $Q$ by greedy volume maximization, and the observation matrix is
$Q_{\mathrm{obs}} = Q[\text{rows}]$. Reconstruction solves

$$
Q_{\mathrm{obs}}\,\theta = y_{\mathrm{obs}},
$$

and the 2-norm condition number of $Q_{\mathrm{obs}}$ is recorded with the orbit. The rows are
*observations*, not parameter pivots: an observed component does not equal a parameter.

## Implementation in MLFCS

* `LatticeFrame` stores $A_s$, $A$, $U$, $U^{-1}$, the motif match and $K_n$, and converts fractional
  coordinates, integer translations and rotations exactly between the two frames. The reduction is
  Minkowski's, and for degenerate lattices (equal lengths, signs, axis permutations) a stable "cell +
  motif" key picks one representative among finitely many signed permutations, so equivalent unimodular
  inputs produce **the same integers**.
* `interactions/algebra/exact.py` provides the modular rank certificate and the saturated integer
  kernel. A rank modulo $p$ never exceeds the rank over $\mathbb Q$, so a full rank at one prime is a
  proof; a deficient verdict is certified once the product of the distinct primes exceeds the Hadamard
  bound of the largest minors. The kernel comes from a Smith normal form decomposition, so it is the
  saturated integer kernel rather than columns divided by their greatest common divisors.
* `interactions/algebra/invariants.py` stacks the stabilizer residual constraints, takes the dimension
  from the certificate, returns the integer basis and **verifies exactly** that the constraint matrix
  kills it.
* The traversal runs in the algebra frame and maps every member back with
  `LatticeFrame.source_labels`, so a reference supercell index still addresses the same physical
  interaction.
* Cartesian rendering has a single entry point, `interactions/algebra/rendering.py`; `frame @ basis`
  appears in no consumer. Fitting design kernels, finite-difference reconstruction, ASR and expansion all
  consume $Q$.
* Realization identifiability ranks the integer realization matrix with the same certificate, with no
  coefficient threshold and no rank tolerance.

## Numerical considerations

* Exact and floating point work is deliberately separated: rank, kernels and stabilizer invariance are
  decided on integers, while Cartesian rendering, QR and observation-row selection use floating point on
  an $O(1)$ numerical basis.
* The observation condition number is stored with each orbit (`observation_condition`). Equivalent
  representations produce bit-identical integer algebra, so it cannot degrade with the shear of the input
  cell; measured values stay below about 5 at fourth order.
* Lattice integers pass an explicit int64 range check before they enter tensor arithmetic; a wider value
  raises `IntegerRangeError` naming the quantity instead of letting NumPy truncate it or a C extension
  raise an unrelated conversion error.
* The rank certificate consumes primes **on demand**: `prevprime` supplies distinct primes and the loop
  stops once their product exceeds the Hadamard bound. It is not an infinite enumeration of the primes;
  when the available primes cannot settle a certificate, `RankCertificateError` is raised. Arbitrary-size
  integers are supported in the rank and kernel layer only, not end to end.

## Validation

* Equivalent representations of one crystal (adjacent, crossed and large shears of cubic, fcc primitive,
  hexagonal and tilted cells, orders two to four) must agree on their integer bases, Cartesian subspaces,
  observation rows, condition numbers, expanded force constants and finite-difference reconstructions.
* Stabilizer constraints are cross-checked against an independent enumeration in the source frame that
  uses a floating point rank oracle.
* Exact rank is checked against SymPy, including zero matrices, deficient matrices, primes that divide a
  witness minor, and entries at $2^{63}$ or $10^{30}$.
* The analytic Morse FC4 oracle and the tutorial cases pin the physical content: rebasing must not change
  a delivered force constant.

## Primitive cell, explicit supercell and one precision

The orbit and interaction algebra runs on the **user-provided** primitive cell, and the reference
supercell is always supplied explicitly: no core path guesses one from a cutoff or builds one
implicitly.  The integer replication relation between the two cells, the mapping of supercell atoms
onto "primitive atom plus integer lattice vector", and the spglib identification of the primitive
symmetry all use one length precision, `symprec` in angstrom; dimensionless matrix elements,
fractional coordinate differences and angles are never compared with it.
`mlfcs.tools.supercell.build_supercell` is an optional convenience tool that the core does not
depend on.
