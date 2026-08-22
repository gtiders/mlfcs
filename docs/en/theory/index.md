---
title: Theory
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Theory

## Motivation

The theory section follows the mathematical path from the crystal potential-energy surface to force constants, periodic interactions, fitting, constraints, and temperature-dependent harmonic models.

## Mathematical object

The quantity is defined on the displacement space of the entire periodic crystal. Atom labels, Cartesian components, and integer translations must be retained together.

## Implementation in MLFCS

MLFCS separates structure mapping, interaction/orbit construction, parameterization, and physical IFCs. Reduced parameters are computational coordinates; outputs recover labelled Taylor force constants.

## Numerical considerations

Finite supercells, truncation, floating-point tolerances, and matrix rank limit recoverable information. Symmetry removes redundancy but cannot create observations.

## Validation

Use analytic models, symmetry transforms, atom permutations, non-diagonal supercells, and material cases. Canonical ordering changes must not change physical observables.
