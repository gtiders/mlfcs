# Rotational sum-rule projections

These examples compare the same FC2 fit before and after physical post-processing:
translational invariance alone, or translational invariance together with the
Born–Huang and Huang conditions. Each fit is an independent task with its own
inputs, script, `fit.log`, and `metrics.json`.

Run each fit and then regenerate the material's phonon-band comparison:

```bash
uv run python MoS2-monolayer/asr/fit.py
uv run python MoS2-monolayer/born-huang-huang/fit.py
uv run --with phonopy --with matplotlib python MoS2-monolayer/plot.py

uv run python graphene/asr/fit.py
uv run python graphene/born-huang-huang/fit.py
uv run --with phonopy --with matplotlib python graphene/plot.py
```

The fitting scripts use the new primitive-cell, supercell, cluster-space,
`FitSystem`, and `ForceConstants` APIs. The ASR path projects the fit onto
translational invariance. The rotational path performs one joint projection
onto ASR, Born–Huang, and Huang conditions. Every `fit.log` is overwritten by
its own task and contains stdout, stderr, and traceback.
