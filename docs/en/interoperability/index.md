---
title: Formats
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Formats

Native HDF5 is the canonical interchange format. It stores primitive/reference structures,
the general supercell matrix, atom mappings, sparse IFCs, units, constraints, and provenance.

Export is a validated view operation. It may reorder atoms, shift the periodic origin, or apply an
integer unimodular basis change, but it must preserve the primitive and supercell translation
lattices. No writer may enlarge, shrink, strain, or redefine a primitive cell.

| Consumer | Format |
|---|---|
| MLFCS and high-order workflows | [Native HDF5](native-hdf5.md) |
| phonopy | [FC2 text](phonopy-text.md) |
| phono3py | [phonopy and phono3py HDF5](phonopy-hdf5.md) |
| ShengBTE | [FC3/FC4 text](shengbte.md) |
| ALAMODE | [FCSXML](alamode.md) |
