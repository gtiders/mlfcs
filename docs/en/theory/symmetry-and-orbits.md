---
title: Symmetry and orbits
audience:
  - advanced
status: stable
code_verified: 4.0.0a5
---

# Symmetry and orbits

## Motivation

Describe the affine space-group action on primitive sites and integer translations, index permutations, stabilizers, and invariant Cartesian tensor bases.

## Mathematical object

The quantity is defined on the displacement space of the entire periodic crystal. Atom labels, Cartesian components, and integer translations must be retained together.

## Implementation in MLFCS

MLFCS separates structure mapping, interaction/orbit construction, parameterization, and physical IFCs. Reduced parameters are computational coordinates; outputs recover labelled Taylor force constants.

## Numerical considerations

Finite supercells, truncation, floating-point tolerances, and matrix rank limit recoverable information. Symmetry removes redundancy but cannot create observations.

## Validation

Use analytic models, symmetry transforms, atom permutations, non-diagonal supercells, and material cases. Canonical ordering changes must not change physical observables.
