# Si FC2 by finite differences

This directory independently calculates second-order force constants for Si
using ASE and the bundled NEP3 model. It needs only the files in this directory
and an installed MLFCS package.

Run from this directory with the repository environment:

```bash
uv run --project ../../../.. --with calorine python run.py
```

The script writes its complete stdout, stderr, and traceback to `run.log`,
overwriting that log on each run. It generates a $4\times4\times4$ supercell
supercell, evaluates the six displacement structures, reconstructs FC2,
projects the result onto translational invariance, and writes:

- `fc2.mlfcs`: native MLFCS force constants;
- `FORCE_CONSTANTS`: phonopy text FC2;
- `force_constants.hdf5`: phonopy HDF5 FC2;
- `SPOSCAR`: the supercell in explicit atom order;
- `metrics.json`: configuration count and ASR projection diagnostics.

The $0.01$ Å step and $7.7237404951$ Å cluster cutoff match the earlier Si
finite-difference example. The four-by-four-by-four supercell gives 28 FC2
parameters and six finite-difference configurations with the current API.

## Potential source

`Si_2022_NEP3_5body.txt` is the Si NEP model from the authors' [nep-data
repository](https://gitlab.com/brucefan1983/nep-data). The model and its
training data are associated with Albert P. Bartók et al., [Machine Learning a
General-Purpose Interatomic Potential for Silicon](https://doi.org/10.1103/PhysRevX.8.041048),
*Physical Review X* **8**, 041048 (2018), and Zheyong Fan et al., [GPUMD: A
package for constructing accurate machine-learned potentials and performing
highly efficient atomistic simulations](https://doi.org/10.1063/5.0106617),
*The Journal of Chemical Physics* **157**, 114801 (2022). Follow the source
repository's license and citation terms.
