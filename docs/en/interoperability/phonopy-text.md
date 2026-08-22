---
title: Phonopy text FC2
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Phonopy text FC2

## Representation boundary

Describe full-supercell dense FC2 text output, target ordering, and validation requirements.

## Before conversion

Specify source primitive/reference, target structure, atom order, units, and tensor-component convention. Validation precedes writing.

## Conversion

Realize canonical primitive IFCs on the target supercell, then apply format-specific folding, densification, block order, or image encoding. Writers do not rerun orbit enumeration.

## Validation

Prefer round trips and actual third-party readers. File size, line order, or representative choice alone does not establish an error.

## Rejection

If a target format cannot represent an order, translation, image, or structure relation, the writer rejects it instead of silently averaging or changing the model.
