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

The Ba cell has 54 primitive atoms, so a full kALDo transport calculation can
have a high memory peak. This example validates IFC loading and records the 162
harmonic frequencies at the Gamma point; it does not calculate thermal
conductivity. kALDo runs on the CPU; this check does not require GPU support.
Install it in an independent environment (the project environment is not
modified):

```bash
uv venv /tmp/kaldo-env --python 3.12
uv pip install --python /tmp/kaldo-env/bin/python kaldo
PYTHONPATH=src /tmp/kaldo-env/bin/python tutorial/Ba8Ga16Ge30/kappa/run.py
```

The transport script uses the MLFCS export API to write VASP/phonopy FC2 text
as `FORCE_CONSTANTS_2ND` and ShengBTE FC3 text as `FORCE_CONSTANTS_3RD`. kALDo
loads these with its `vasp-sheng` reader and uses `primitive.vasp` as the
primitive-cell structure. The IFC inputs are staged in a temporary directory;
the Gamma-point log and JSON summary are written under `kappa/`. A bulk
conductivity requires a converged Brillouin-zone mesh and a memory-bounded
transport run.

The 300 K training snapshots and structures come from the hiPhive
Ba8Ga16Ge30 clathrate thermal-conductivity example. The upstream case uses a
force-constant potential with FC2/FC3/FC4 cutoffs of 5.4/4.35/4.35 Å and a
maximum 2-body interaction. This migration fits only FC2 and FC3 from the
provided snapshots; it does not read or redistribute the upstream FCP. Please
retain the upstream case's attribution and licensing terms when reusing the
data.
