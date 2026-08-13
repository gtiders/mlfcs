# Differences between the previous and current MLFCS implementations

English | [中文](OLD_NEW_COMPARISON_ZH.md)

This document records compatibility and intentional numerical changes. It is not a second user
guide; current usage belongs in the root README and the external-workflow guide.

## Summary

| Area | Previous implementation | MLFCS 3.0 |
|---|---|---|
| Interface | Order-specific CLI and files | Python API with ASE objects |
| Orders | Separate FC3 and FC4 paths | One `order >= 2` pipeline |
| Force source | Integrated workflow assumptions | User calculator or external forces |
| Identity | Implicit filename/position conventions | IDs, plan hash, order metadata |
| Storage | Mostly dense, order-specific | Sparse clusters and lazy materialization |
| ASR | Relative compensation; incorrect FC4 sum | Strict parameter-space constraint |
| Acceleration | CPU-centric dense transforms | JAX CPU/GPU tensor kernels and sparse solvers |
| Output | Separate order-specific writers | Explicit, extensible `format` routing |

## Geometry, atom order, and cutoffs

MLFCS 3.0 stores each supercell atom's primitive index and cell translation and uses one canonical
cell-major order. Grouped order remains available at VASP, phonopy, and phono3py boundaries.
Negative cutoffs select one-based neighbor shells with legacy-compatible distance tolerances.
The program distinguishes the finite supercell's maximum enumerable shell from the shell and
radius selected by the user.

Cluster enumeration and default ShengBTE output now share one joint periodic-image predicate,
giving a symmetry-closed support. The optional `compatibility="thirdorder"` writer reproduces
the previous secondary image filtering and block order. This switch changes only the ShengBTE
representation boundary, not reconstruction.

## Symmetry and finite differences

The current orbit engine combines space-group mappings, all index permutations, Cartesian tensor
rotations, and stabilizer constraints for any order. This avoids fixed FC3/FC4 branches and fixes
fragile multi-species mappings that could confuse same-position periodic images or apply an atom
permutation without the matching Cartesian action.

Recursive central differences generate the force derivative for any order. Equivalent
displacements are deduplicated, but the new irreducible representatives need not appear in the
same positional order as old `3RD.POSCAR.*` files. Old forces must therefore be matched through a
verified plan, never fed positionally into a new plan.

## Sparse reconstruction and memory

The previous order-specific flow could allocate dense tensor actions and full force-constant
arrays early. The current result retains only symmetry-generated cluster tensors. Dense export is
lazy and warns when its estimated size exceeds the advisory budget. Matrix-free rotations, small
Gram matrices, sparse LSMR, contiguous arrays, and serial calculator evaluation reduce peak
memory, while generic sparse HDF5 remains usable for higher orders.

## Acoustic sum rule

For every order, translational invariance requires the sum over one atom index to vanish while all
other indices remain fixed. The old FC3 relative-weight adjustment did not solve a strict
constraint and could amplify large components; the old FC4 logic summed two atom axes together,
which is not the same physical rule. MLFCS 3.0 constructs `A p = 0` in the independent orbit basis
and projects with a Gram null space plus LSMR, or sparse LSMR directly for large systems.

Consequently raw results should be compared when validating finite differences, and constrained
results should be compared only with another explicitly constrained reference. A difference from
the old compensated output is intentional when the old operation changes the physical problem.

## Formats and supported orders

Order 2 and all higher orders use the same computation object. Generic sparse HDF5 supports every
order; NumPy requires materialization. Phonopy text/HDF5 are provided for FC2, phono3py HDF5 for
FC3, and ShengBTE text for FC3/FC4. Higher-order exchange should use sparse HDF5 because there is
no generally adopted ShengBTE text format above FC4.

The native SSCHA module is separate from the finite-difference workflow but reuses the MLFCS FC2
orbit parameterization and streamed-Gram fitter. It samples quantum or classical harmonic
ensembles directly in compact-FC2 q space without phonopy or symfc at runtime; it does not
implement explicit FC3 bubble or FC4 loop diagrams.

## Compatibility policy

MLFCS preserves physical tensor semantics, deterministic sow/reap identity, documented atom
orders, neighbor-shell behavior, and supported external formats. It does not preserve incorrect
ASR mathematics, undocumented positional interchange between different plans, or unnecessary
dense allocation. Numerical evidence and CI thresholds are recorded in the
[validation guide](VALIDATION.md).
