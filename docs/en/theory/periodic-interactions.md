---
title: Periodic geometry and IFCs
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# Periodic geometry and IFCs

`StructureRelation` maps a primitive to one concrete reference without assuming diagonal repeats
or cell-major atom order. Its HNF-backed quotient is used only to realize exact lattice labels in
that finite reference. Interaction discovery itself is performed in the infinite primitive
lattice.

The native sparse model stores lattice-labelled entries:

```text
sites                        (K, order)
translations  (K, order - 1, 3)
tensors                      (K, 3, ..., 3)
```

The integer translations are exact primitive-lattice vectors, not finite-supercell residues.
Consequently Fourier phases use them directly and the same IFC can be realized in any verified
integer supercell. Residue reduction occurs only at a finite calculation or output boundary;
format-specific image rules remain confined to their writer.
