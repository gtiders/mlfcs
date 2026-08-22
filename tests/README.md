# Test organization

[中文](README_ZH.md)

Tests validate current public behavior, mathematical constraints, and independent scientific
references. They do not use the legacy MLFCS implementation as an oracle.

## Layers

- `unit/` contains fast geometry, finite-difference, reconstruction, ASR, and I/O tests.
- `integration/` exercises public `sow()`, `reap()`, direct ASE calculators, and SSCHA.
- `reference/` contains independent scientific baselines and fixed external data.

`reference/analytic/Morse_FCC_FC4/` uses ASE Morse forces and an independently JAX-differentiated
pair energy. It checks the analytic FCC equilibrium, FC4 values, and second-order step
convergence without another force-constant fitter or ASR.

`reference/phono3py/AlN_FC3/` compares raw and ASR-constrained AlN FC3 with phono3py, validates
the hiphive representation adapter, and checks fixture provenance. The separate
`reference/phonopy/AlN_FC2/` baseline checks full FC2 from the same training data and potential.

`reference/phono3py/K4As4Pt2_FC23/` covers a multicomponent 2x2x3 model at the maximum MIC cutoff:
FC2, FC3, ASR, a packaged pypolymlp potential, official HDF5 readers, and faithful ShengBTE
roundtrips.

`reference/shengbte/Si_FC3/` freezes the 3x3x3, `cutoff=-6` VASP sow order and reconstructs
ShengBTE FC3 from 168 external force calculations. The external comparison explicitly uses
`compatibility="thirdorder"`; default faithful support is tested separately.

## Naming and execution

Chemical formulas retain standard capitalization (`AlN`, `Si`, `NaCl`). Project and algorithm
names use their official spelling (`ASR`, `phono3py`, `hiphive`). Scientific reference tests must
run serially.

```bash
uv run pytest -m "not reference"
uv run pytest tests/reference/analytic/Morse_FCC_FC4/test_morse_fc4.py
uv run pytest tests/reference/phono3py/AlN_FC3
uv run pytest tests/reference/phonopy/AlN_FC2
uv run pytest tests/reference/phono3py/K4As4Pt2_FC23
uv run pytest tests/reference/shengbte/Si_FC3
```

Fixture generation is a maintainer operation documented in [`reference_tools/`](../reference_tools/README.md);
it is not part of ordinary CI.
