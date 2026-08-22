---
title: Native HDF5 v3
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Native HDF5 v3

## Representation boundary

Native HDF5 v3 stores the primitive structure and exact real-space IFC rows: primitive sites, integer primitive-lattice translations, and Cartesian tensors. It contains no source-supercell mapping. After reading, the same IFCs can be realized in any verified integer supercell. Files from older schemas are rejected with an unsupported-schema error. There is no migration reader that guesses old atom-order semantics.

## Before conversion

Specify source primitive/reference, target structure, atom order, units, and tensor-component convention. Validation precedes writing.

## Conversion

Realize canonical primitive IFCs on the target supercell, then apply format-specific folding, densification, block order, or image encoding. Writers do not rerun orbit enumeration.

## Validation

Prefer round trips and actual third-party readers. File size, line order, or representative choice alone does not establish an error.

## Rejection

If a target format cannot represent an order, translation, image, or structure relation, the writer rejects it instead of silently averaging or changing the model.
