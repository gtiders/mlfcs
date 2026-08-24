---
title: ASE Calculator
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# ASE Calculator

`MLFCSCalculator` interprets stored canonical Taylor force constants as a fixed-cell polynomial
potential and exposes standard ASE relative-energy and atomic-force properties. It performs no
fitting, does not load JAX, and does not require the training data.

## Signatures

```python
MLFCSCalculator(
    force_constants: ForceConstants,
    *,
    reference: Atoms | None = None,
    maximum_displacement: float | None = None,
)

MLFCSCalculator.from_hdf5(
    source: str | Path,
    *,
    reference: Atoms | None = None,
    maximum_displacement: float | None = None,
) -> MLFCSCalculator
```

| Parameter | Meaning |
|---|---|
| `force_constants` | `ForceConstants` containing canonical Taylor IFCs of order two or higher. Explicit non-Taylor metadata is rejected. |
| `source` | An MLFCS native HDF5 v3 file. |
| `reference` | The fixed explicit supercell used by the calculator. The current relation reference is used by default; an HDF5 file defaults to its primitive cell. |
| `maximum_displacement` | Optional positive warning threshold in Å. Evaluation continues without clipping when it is exceeded. |

## Energy and force semantics

The calculator evaluates

$$
\Delta E(\mathbf u)=
\sum_{n\ge2}\frac{1}{n!}\Phi^{(n)}\mathbf u^n,
$$

and

$$
\mathbf F=-\frac{\partial\Delta E}{\partial\mathbf u}.
$$

It defines $E_0=0$ and FC1 $=0$. The reported energy is relative to the reference rather than an
absolute electronic-structure energy, and both energy and force vanish at the reference.

```python
from ase.io import read
from mlfcs import MLFCSCalculator

reference = read("supercell.vasp")
atoms = reference.copy()
atoms.positions[0, 0] += 0.01
atoms.calc = MLFCSCalculator.from_hdf5("mlfcs.h5", reference=reference)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Fixed boundaries

- Atom count, species order, cell, and PBC must match the construction reference.
- Periodically equivalent wrapped coordinates are handled through MIC displacements; atoms are not reordered.
- Exact-$R$ IFCs may be realized into another legal integer target supercell at construction.
- Only `energy` and `forces` are implemented; stress, virial, absolute energy, and cell strain are not.
- The calculator evaluates the stored Taylor polynomial, not fitting-basis parameters or omitted reference forces.
