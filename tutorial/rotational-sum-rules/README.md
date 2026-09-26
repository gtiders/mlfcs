# Rotational sum-rule projections

These examples compare the same FC2 fit before and after physical post-processing:
translational invariance alone, or translational invariance together with the
Born–Huang and Huang conditions. Each fit is an independent task with its own
inputs, script, `fit.log`, native `fc-fit.mlfcs` model, and Phonopy
`fc2-phonopy.txt` export.

Run each fit and then regenerate the material's phonon-band comparison:

```bash
uv run python MoS2-monolayer/asr/fit.py
uv run python MoS2-monolayer/born-huang-huang/fit.py
uv run --with phonopy --with matplotlib python MoS2-monolayer/plot.py

uv run python graphene/asr/fit.py
uv run python graphene/born-huang-huang/fit.py
uv run --with phonopy --with matplotlib python graphene/plot.py
```

The fitting scripts build the cluster space directly from the primitive ASE
structure, infer the supercell relation from the supplied cells, and use
`FitSystem` and `ForceConstants`. The ASR path projects the fit onto
translational invariance. The rotational path then projects onto the
Born–Huang and Huang conditions. Every `fit.log` is overwritten by its task
and contains stdout, stderr, and traceback. The phonon plots are regenerated
as `phonon-bands.png` in each material directory.
