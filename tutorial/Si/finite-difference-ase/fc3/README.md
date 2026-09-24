# Si FC3 by finite differences

This directory independently calculates third-order force constants for Si
using ASE and the bundled NEP3 model. It needs only the files in this directory
and an installed MLFCS package. It does not read the FC2 tutorial's output.

Run from this directory with the repository environment:

```bash
uv run --project ../../../.. --with calorine python run.py
```

The script writes its complete stdout, stderr, and traceback to `run.log`,
overwriting that log on each run. It generates its own $4\times4\times4$
supercell, evaluates the 140 displacement structures, reconstructs
FC3, projects the result onto translational invariance, and writes:

- `fc3.mlfcs`: native MLFCS force constants;
- `fc3.hdf5`: phono3py FC3 HDF5;
- `FORCE_CONSTANTS_3RD`: ShengBTE FC3 text;
- `SPOSCAR`: the supercell in explicit atom order;
- `metrics.json`: configuration count and ASR projection diagnostics.

The $0.01$ Å step and $7.7237404951$ Å cluster cutoff match the earlier Si
finite-difference example. The four-by-four-by-four supercell gives 711 FC3
parameters and 140 finite-difference configurations with the current API.

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
