# Si finite differences

This tutorial calculates second- and third-order force constants for silicon
with the bundled NEP3 model and ASE. Each order is a self-contained workflow:
run either directory independently.

- [FC2](fc2/README.md)
- [FC3](fc3/README.md)
- [FC3 zero-displacement extrapolation](fc3-extrapolated/README.md)
- [phono3py thermal conductivity](phono3py-kappa/README.md)
- [phono3py conductivity with extrapolated FC3](phono3py-kappa-fc3-extrapolated/README.md)

Both workflows use a four-by-four-by-four supercell and a 0.01 Å
central finite difference. They construct their own primitive cell, cluster
space, supercell mapping, calculator, and force constants. The FC3 calculation
does not consume the FC2 result.

The thermal-conductivity workflow consumes the HDF5 outputs from both FC2 and
FC3, and uses the FC3 workflow's explicit supercell as the phono3py input.

The NEP model and its training data are distributed by the authors of the
potential. See each order's README for source and citation information. MLFCS
does not train or modify the model.
