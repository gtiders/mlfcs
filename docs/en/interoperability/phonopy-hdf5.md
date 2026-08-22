---
title: Phonopy and phono3py
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Phonopy and phono3py

## Representation boundary

FC2 phonopy text is a dense representation and therefore requires a compatible target supercell. phono3py HDF5 is an external schema and should be written only when the target phono3py version and its companion supercell are known. Native MLFCS HDF5 remains the lossless source for all orders. Use the downstream program's own primitive and supercell whenever possible. Never compare raw dense array indices across two independently chosen supercell orderings.

## Before conversion

Specify source primitive/reference, target structure, atom order, units, and tensor-component convention. Validation precedes writing.

## Conversion

Realize canonical primitive IFCs on the target supercell, then apply format-specific folding, densification, block order, or image encoding. Writers do not rerun orbit enumeration.

## Validation

Prefer round trips and actual third-party readers. File size, line order, or representative choice alone does not establish an error.

## Rejection

If a target format cannot represent an order, translation, image, or structure relation, the writer rejects it instead of silently averaging or changing the model.
