---
title: "hiPhive Comparison: Initialization and Representation"
audience:
  - advanced
  - developer
status: research
code_verified: 4.0.0a6
---

# hiPhive Comparison: Initialization and Representation

This note compares the stage that runs *before* a fit or a finite-difference solve -- structure,
symmetry, cluster discovery, invariant basis, realizability -- and the representation the fit then
uses, between MLFCS 4.0.0a6 and hiPhive 1.5.  The hiPhive statements come from reading its source for
this note, so every reference below is a file and line of that release.

## What the two implementations share

Both do the symmetry algebra in the lattice (scaled) frame, and that is the decisive common choice:

- spglib rotations are integer matrices there.  hiPhive publishes them as
  `cluster_space.rotation_matrices` (`cluster_space.py:164`), MLFCS as
  `PrimitiveSymmetryOperations.rotations`.
- The first basis is the label-symmetric indicator basis: `init_ets_from_label_symmetry`
  (`core/eigentensors.py:47`) is the same $0/1$ construction as `label_symmetric_basis`.
- The parameterization is a lattice-frame basis, and Cartesian components appear only at the
  boundary: hiPhive converts through `rotation_to_cart_coord(R, cell)` (`core/tensors.py:38`) when it
  assembles supercell force constants, MLFCS through one deterministic
  $K = (\text{cell}^T)^{\otimes\text{order}}$ map when a consumer needs physical tensors.  The two
  formulas are the same map, which is why the frame calibration in
  [orbit-search-thresholds.md](orbit-search-thresholds.md) could be pinned against spglib's own
  Cartesian rotations.

## Where the initialization stages differ

| Step | hiPhive 1.5 | MLFCS 4.0.0a6 |
| --- | --- | --- |
| Cluster discovery | Enumerates *all* clusters in the supercell up to the cutoffs, then reduces them: `get_orbits(cluster_list, atom_list, rotation_matrices, ...)` (`core/orbits.py:18`) is handed a finished cluster list | Generator-closure orbit search: seeds from `iter_primitive_candidates` are closed under the generator set with Schreier stabilizers and a canonical anchor (`traverse_indexed_orbit`), so clusters are never enumerated wholesale |
| Invariant basis | Iterative reduction per symmetry operation: sparse constraint matrix, exact rational null space by `SparseMatrix.rref_sparse` / `nullspace` (`core/utilities.py:25`), then `renormalize_to_integer` (`core/eigentensors.py:112`) | Constraints come from the stabilizer set only (Schreier generators, fewer than the whole group); integer Gram in `int64`; dimension from a rank modulo a large prime; kernel solved in float and verified column by column over the integers, then divided by column gcds |
| Rank / exactness | Symbolic elimination decides the null space | Both directions of the rank are certified by primes; see `certified_rank`.  A prime can only underestimate the rank over $\mathbb{Q}$, so a full rank is a proof and a deficient verdict adds primes until their product exceeds the exact Hadamard bound of the largest minors |
| Parameterization | The eigentensors *are* the parameters; there is no pivot concept | The integer lattice basis is the parameterization and the fitted parameters are its coefficients, plus `pivots`: the Cartesian component rows a finite-difference plan observes and that determine those coefficients |
| Identifiability of a reference | No equivalent check: an under-determined reference is handled by ridge regularisation (`enforce_rotational_sum_rules` fits with `Ridge(alpha=...)`) | `validate_realization_identifiability` ranks the integer lattice realization matrix per connected component and raises `InteractionAliasingError`, naming the conflicting clusters and both remedies |
| Sum rules | Rotational sum rules and the acoustic sum rule are post-fit projections | The acoustic sum rule enters the fit as an equality-constrained least-squares system; rotational rules stay a post-fit projection |
| Cutoffs | Explicit distances, chosen by the caller | Explicit distances or negative shell indices, chosen by the caller (the reference-resolved `cutoff=None` mode was removed, so the two APIs now match in style) |
| Failure mode | A folded reference yields a regularised solution | A folded reference is rejected while the orbit space is realized |

The practical consequence is the failure mode: with hiPhive an insufficient reference is *silent*,
because the regulariser absorbs it; with MLFCS it is an error that names the clusters to fix, which is
what allowed the reference-resolved cutoff to be deleted in favour of that check.

## Measured cost

The invariant kernel at order 4, per orbit, measured on this machine:

| Route | Cost |
| --- | --- |
| A literal port of hiPhive's recipe (its `SparseMatrix` null space, iterated over the point group) | 30–70 ms |
| MLFCS (`invariant_kernel`: integer Gram, modular rank, verified kernel) | 0.4–2 ms |

Certifying an exact rank over the whole order-4 space of Si (12 Grams) costs 361 ms, the two-prime
fast path 42 ms, and a rejected reference is reported in 1–48 ms.  The gap comes from what each route
does per symmetry: symbolic elimination with growth, against integer arithmetic whose decisions are
either proofs modulo a prime or verified over the integers.

## Measured agreement on the shipped NaCl case

The long-range electrostatics tutorial runs both codes on the same upstream data (8-atom rocksalt
cell, $4\times4\times4$ reference, two finite-displacement configurations, 11 Å FC2 cutoff).
Committed metrics:

| Quantity | MLFCS | hiPhive |
| --- | --- | --- |
| Parameters (23 orbits in MLFCS) | 76 | 74 |
| Short-range force RMSE (eV/Å) | $1.484222611000188\times10^{-5}$ | $1.4842226109978366\times10^{-5}$ |
| Total-force RMSE (eV/Å) | $2.651939354365933\times10^{-5}$ | $2.6519393543682293\times10^{-5}$ |
| Relative force error (short range) | $0.024691623576418457$ | $0.024691623576379346$ |
| Restored FC2, relative Frobenius difference | $4.27\times10^{-10}$ | -- |
| Phonon band difference against phonopy+NAC, short-force fit (THz, max) | $0.11648212566859484$ | $0.1164821254595414$ |

Reproduce with `prepare.py`, `fit_mlfcs.py` and `run_hiphive.py` in that tutorial directory; hiPhive 1.5
currently needs NumPy below 2.5 through numba, which is why its command builds a throwaway
environment.

The remaining difference to explain is the parameter count, 76 against 74, while the fitted FC2 agrees
to $4.3\times10^{-10}$ relative: the two parameterizations are equivalent on this data but are not the
same set of numbers.

## What was deliberately not taken from hiPhive

- The symbolic route: exact, but 30–70 ms per orbit at order 4 (13–44 s for the order-4 spaces in this
  repository) against 0.096 s for the whole invariants stage before the lattice-frame work.
- Handling an under-determined reference by regularisation: it produces an answer instead of an error,
  and the reference is exactly what a caller can fix.
- Its cluster enumeration order: quadratic in supercell clusters rather than in orbit size.
