# NaCl: long-range dipole forces

This example adapts hiPhive's `advanced_topics/long_range_forces` NaCl data to
MLFCS. The input files `NaCl_unitcell.xyz`, `supercells_with_forces.xyz`, and
`BORN` come from hiPhive; its MIT license is included in `LICENSE`. The two
displaced 512-atom structures carry their precomputed forces, so no external
calculator or hiPhive installation is needed.

Run from the repository root:

```bash
.venv/bin/python tutorial/NaCl/fit.py
.venv/bin/python tutorial/NaCl/plot.py
```

`fit.py` builds a 2-atom primitive cluster space with an 11 Å FC2 cutoff. It
first fits the supplied forces directly. It then computes the fixed-supercell
dipole Ewald forces from the Born charges and dielectric tensor in `BORN`,
explicitly fits $F^{\mathrm{short}}=F^{\mathrm{data}}-F^{\mathrm{Ewald}}$, and
exports $\Phi^{\mathrm{total}}=\Phi^{\mathrm{short}}+\Phi^{\mathrm{Ewald}}$.
Both fitted models receive an acoustic-sum-rule projection before export; the
Ewald FC2 already satisfies the acoustic sum rule by construction.
The two native `.mlfcs` files contain the direct and short-range models;
`fc2-direct.hdf5` and `fc2-corrected.hdf5` are reproducible Phonopy exports.
The full fitting output, including any traceback, is overwritten in `fit.log`.

`plot.py` uses Phonopy to obtain a finite-displacement reference from the same
forces and Seekpath for the band path. It applies Phonopy's non-analytic
correction (NAC) to the reference and both fitted FC2 arrays and writes
`phonon-bands.png`. The upper panel isolates the effect of NAC; the lower panel
compares direct fitting against explicit long-range subtraction and addition.
The HDF5 force constants alone do not contain the NAC parameters: `BORN` must
still be supplied to Phonopy when calculating polar phonons.
