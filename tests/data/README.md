# Migration fixtures

`kernel_constraints.npz` stores the 165 constraint matrices from the G-section
profiling corpus. Matrices were constructed with the original NumPy tensor
contraction/permutation formula and duplicate rows removed. `names`, `orders` and
`ranks` identify each record; `a000` through `a164` contain integer matrices.

The source profiling record is `/tmp/mlfcs-kernel-profile.json`, referenced in
`docs/numba-integer-audit.md`. Coverage includes FC2–FC6, seven crystal systems,
Si, Mg, shear 7/31, graphene, MoS2, K4As4Pt2 and Ba8Ga16Ge30. Tests independently
check exact annihilation, recorded rank and all Smith invariant factors of the
basis equal to one. Random matrices additionally compare full lattice HNFs
against the original SymPy kernel algorithm.

`force_constants_v1.mlfcs` is a trusted test-only native pickle written by the
original Python 4.5.1 implementation for a simple cubic Ar FC2 model. It is retained only to verify that version 6 rejects old files before unpickling.

`force_constants_v3.mlfcs` contains the same physical model converted through
neutral ndarray state, with identical model and cluster-space fingerprints. It
exercises version-3 loading and immutable/int64 normalization.

`force_design_reference.npz` records original Python 4.5.1 Si FC2/FC3 design
matrices, six displaced snapshots, synthetic forces and representative fitted
tensors. It covers the downstream design and fitting path independently of the
new backend.

`symmetry_reference.json` freezes the validated pre-migration backend outputs
for Si FC2/FC3 and Mg FC3: representative labels, orbit images, actions,
saturated lattice bases, Cartesian bases and folded ranks for two supercells.
The historical implementation and validation are described in the mathematical
audit. Its migration stash has been removed after validation; tests retain these
records and consume them without a compiled extension.
