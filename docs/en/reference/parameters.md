# Units and parameters

| Quantity | Unit or convention |
|---|---|
| Cell, positions, displacements | Å |
| Forces | eV/Å |
| Order-`n` IFC | eV/Åⁿ |
| Positive cutoff | Å radius |
| Negative cutoff | neighbor-shell index |
| `None` cutoff | maximum radius enumerable by the reference supercell |
| SCPH tolerance | RMS THz frequency change |

JAX kernels use 64-bit floating point. `mixing` is a numerical relaxation coefficient and is not
a physical parameter. `tolerance` is a stopping criterion; it does not zero force constants or
alter the fitted support.
