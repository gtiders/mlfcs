# Si thermal conductivity with phono3py

This workflow calculates the lattice thermal conductivity of Si from the
neighboring [`fc2`](../fc2/README.md) and [`fc3`](../fc3/README.md)
finite-difference workflows. Run those two workflows first. This directory
then consumes their `SPOSCAR`, FC2 HDF5, and FC3 HDF5 outputs without
recomputing force constants.

Run from this directory:

```bash
uv run --project ../../../.. --python 3.12 --with phono3py==4.4.0 --with phonopy==4.4.0 --with phonors==0.3.0 python run.py
```

The input to phono3py is the explicit 128-atom supercell in `../fc3/SPOSCAR`.
The supercell matrix is the identity because that file already describes the
supercell; `primitive_matrix="auto"` asks phono3py to identify its primitive
cell. This avoids passing a separately supplied primitive cell or assuming
that the supercell itself is primitive.

The calculation uses a $10\times10\times10$ reciprocal mesh, isotope
scattering, and temperatures from 300 K through 900 K in 100 K increments.
`run.log` captures stdout, stderr, and any traceback. The phono3py HDF5 result
and `thermal-conductivity.json` are written here.

The numerical results depend on the phono3py version and its transport
settings. The JSON records the mesh, temperatures, input paths, phono3py
version, and the generated HDF5 filename; the HDF5 file is the authoritative
full conductivity output.

With phono3py 4.4.0, the calculated diagonal conductivity components were:

| Temperature (K) | $\kappa_{xx}$ | $\kappa_{yy}$ | $\kappa_{zz}$ |
|---:|---:|---:|---:|
| 300 | 84.368 | 84.368 | 84.368 |
| 400 | 61.190 | 61.190 | 61.190 |
| 500 | 48.292 | 48.292 | 48.292 |
| 600 | 39.990 | 39.990 | 39.990 |
| 700 | 34.169 | 34.169 | 34.169 |
| 800 | 29.849 | 29.849 | 29.849 |
| 900 | 26.510 | 26.510 | 26.510 |

The tensor components are in W/m-K. The full-precision values are in
`thermal-conductivity.json` and the phono3py HDF5 file.
