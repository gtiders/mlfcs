---
title: ShengBTE
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# ShengBTE

## Representation boundary

ShengBTE FC3 and FC4 text are block-oriented dense views. The writer expands the sparse lattice- labelled IFC into the target reference order and validates the translation lattice first. It does not change the physical IFC support or choose a new primitive cell. For arbitrary-q interpolation, periodic-image choices belong to the common geometry layer; they are not reconstructed from a fixed 27-image box by the core IFC model.

## Before conversion

Specify source primitive/reference, target structure, atom order, units, and tensor-component convention. Validation precedes writing.

## Conversion

Realize canonical primitive IFCs on the target supercell, then apply format-specific folding, densification, block order, or image encoding. Writers do not rerun orbit enumeration.

## Validation

Prefer round trips and actual third-party readers. File size, line order, or representative choice alone does not establish an error.

## Rejection

If a target format cannot represent an order, translation, image, or structure relation, the writer rejects it instead of silently averaging or changing the model.
