---
title: Native HDF5 v3
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Native HDF5 v3

Native HDF5 v3 stores the primitive structure and exact real-space IFC rows: primitive sites,
integer primitive-lattice translations, and Cartesian tensors. It contains no source-supercell
mapping. After reading, the same IFCs can be realized in any verified integer supercell.

Files from older schemas are rejected with an unsupported-schema error. There is no migration
reader that guesses old atom-order semantics.
