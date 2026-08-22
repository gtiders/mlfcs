---
title: Phonopy and phono3py
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Phonopy and phono3py

FC2 phonopy text is a dense representation and therefore requires a compatible target supercell.
phono3py HDF5 is an external schema and should be written only when the target phono3py version
and its companion supercell are known. Native MLFCS HDF5 remains the lossless source for all orders.

Use the downstream program's own primitive and supercell whenever possible. Never compare raw dense
array indices across two independently chosen supercell orderings.
