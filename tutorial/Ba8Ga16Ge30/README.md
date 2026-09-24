# Ba8Ga16Ge30 effective force constants and thermal conductivity

This tutorial migrates the 300 K fitting task from the former `T300K/`
directory. The temperature is a property of the supplied NVE snapshots, so the
case lives directly in this material directory rather than being named after
the temperature.

The fit uses 101 frames of the 432-atom $2\times2\times2$ supercell, a
54-atom primitive cell, FC2/FC3 cutoffs of 5.4/4.35 Å, and a maximum body
order of 2 for both orders. It builds the reusable `FitSystem`, solves its
column-scaled normal equations with MINRES, then projects the resulting force
constants onto translational invariance. `fit.log` is overwritten with the
complete stdout, stderr, and traceback from each fitting run.

Run the fit from this directory:

```bash
uv run python fit.py
```

Then calculate 300 K lattice thermal conductivity using phono3py, isotope
scattering, and a $7\times7\times7$ mesh. All transport inputs and outputs are
kept in the `kappa/` subdirectory:

```bash
uv run --with phono3py python kappa/run.py
```

The transport script exports complete FC2/FC3 HDF5 from the saved MLFCS model
and passes the full-supercell FC3 array, $(432,432,432,3,3,3)$, directly to
phono3py. No compact FC3 file is generated. phono3py identifies the primitive
cell automatically from the supplied supercell. The transport log, JSON
summary, complete IFC files, and conductivity HDF5 are all written under
`kappa/`.

The 300 K training snapshots and structures come from the hiPhive
Ba8Ga16Ge30 clathrate thermal-conductivity example. The upstream case uses a
force-constant potential with FC2/FC3/FC4 cutoffs of 5.4/4.35/4.35 Å and a
maximum 2-body interaction. This migration fits only FC2 and FC3 from the
provided snapshots; it does not read or redistribute the upstream FCP. Please
retain the upstream case's attribution and licensing terms when reusing the
data.
