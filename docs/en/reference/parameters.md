# Units and parameters

| Quantity | Unit or convention |
|---|---|
| Cell, positions, displacements | Å |
| Forces | eV/Å |
| Order-`n` IFC | eV/Åⁿ |
| Positive cutoff | Å radius |
| Negative cutoff | neighbor-shell index |
| `None` cutoff | largest periodic-image-unambiguous exact-$R$ radius of the current reference, with a $0.01$ Å margin below the first ambiguous boundary |
| SCPH tolerance | RMS THz frequency change |

JAX kernels use 64-bit floating point. `mixing` is a numerical relaxation coefficient and is not
a physical parameter. `tolerance` is a stopping criterion; it does not zero force constants or
alter the fitted support.

`cutoff=None` is neither an infinite interaction range nor the complete periodized finite-cell
FC2 used by ALAMODE or phonopy. It selects the largest support on which the current source
reference does not include two periodic images of the same atom pair. Polar crystals with a
long-ranged dipole tail still require source-supercell convergence or an analytic long-range
electrostatic separation.
