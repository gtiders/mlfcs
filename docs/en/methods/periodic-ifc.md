# Periodic geometry and IFCs

The shared `StructureRelation` and `PeriodicIndex` map primitive sites and translation residues
without assuming diagonal repeats or cell-major atom order. `PeriodicGeometry` uses general MIC,
Minkowski-reduced searches, and all tied nearest images.

The native sparse model stores lattice-labelled entries:

```text
sites                        (K, order)
translations  (K, order - 1, 3)
tensors                      (K, 3, ..., 3)
```

Residues are physical finite-supercell labels. For arbitrary-q Fourier interpolation, the SCPH
path resolves them through all degenerate Wigner-Seitz images with equal phase weight. Writers do
not independently infer atom order or periodic geometry.
