# Si FC3 zero-displacement extrapolation

This workflow repeats the FC3 finite differences at five displacement lengths
and extrapolates the derivative to zero displacement. It uses the same Si
primitive cell, NEP3 potential, $4\times4\times4$ supercell, and cluster-space
cutoff as [`../fc3`](../fc3/README.md). It writes an FC3 HDF5 file for the
neighboring phono3py transport workflow.

Run from this directory:

```bash
uv run --project ../../../.. --with calorine python run.py
```

The displacement values are $0.010$, $0.015$, $0.020$, $0.025$, and
$0.030$ Å. The current API combines these estimates by polynomial
zero-displacement extrapolation in the squared step length. `run.log` captures
stdout, stderr, and any traceback. Outputs include native MLFCS force
constants, phono3py FC3 HDF5, ShengBTE text, the generated supercell, and
`metrics.json` with the ASR and extrapolation diagnostics.

The calculation used 700 force configurations (140 per displacement). The
ASR projection reduced the relative residual from $3.83\times10^{-5}$ to
$2.84\times10^{-16}$. The extrapolated FC3 coefficient vector differs from
the ASR-projected $0.020$ Å result by $6.56\times10^{-4}$ in relative $L_2$
norm.
