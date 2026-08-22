---
title: Structures and downstream software
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Structures and downstream software

The primitive cell is an explicit input. A reference supercell may be generated from an integer
3x3 matrix or supplied directly in any atom order. MLFCS records the primitive-site label and
integer translation for every reference atom.

Before calculating, obtain the structures from the program that will consume the result when
possible. This avoids mismatched conventions in phonopy, phono3py, ShengBTE, and ALAMODE.
Equivalent primitive and supercell representations can be supplied at export, but the primitive
volume, atom count, translation lattice, and the supercell's number of primitive cells cannot change.

An integer unimodular basis change changes lattice-vector coordinates without changing the lattice
or volume. It is a representation change, not a new supercell.
