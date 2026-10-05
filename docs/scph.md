# SCPH

`SCPH` performs a static quartic-loop self-consistent phonon calculation. It starts from a `ForceConstants` model containing FC2 and FC4, constructs the irreducible reciprocal stars for a chosen mesh, and iteratively updates the effective FC2 at each temperature. FC3 may also be present in the model, but it does not enter this static loop correction.

## Construct `SCPH`

```python
from mlfcs.phonon import SCPH

scph = SCPH(model, mesh, statistics="quantum", time_reversal=True)
```

| Parameter | Meaning |
| --- | --- |
| `model` | `ForceConstants` on one cluster space, containing both orders 2 and 4. The FC4 block supplies the loop correction; FC2 supplies the bare harmonic model. |
| `mesh` | `QGrid`, a positive integer diagonal mesh such as `(4, 4, 6)`, or a nonsingular 3×3 integer supercell matrix. Reduction uses the mass-preserving subgroup of the primitive symmetry. |
| `statistics` | Modal covariance statistics: `"quantum"` (default) or `"classical"`. |
| `time_reversal` | Whether reciprocal stars include time-reversal pairing; default `True`. |

The constructor exposes `model`, `stars`, and `statistics`. Its reciprocal grid is exact integer data; the star representatives are the points at which the frequency and covariance kernels are evaluated.

Masses come directly from the readonly `model.cluster_space.masses` assignment. Different masses on structurally equivalent sites are supported: only mass-preserving operations enter the reciprocal stars. See [harmonic frequencies and masses](harmonic-api.md) for changing masses without rebuilding the cluster space.

## Run one temperature

```python
result = scph.run(
    300.0,
    mixing=0.2,
    tolerance=1e-9,
    max_iterations=200,
)
print(result.converged, result.iterations, result.minimum_mode_thz)
```

`run(temperature, *, start=None, mixing=0.2, tolerance=1e-9, max_iterations=200)` accepts:

| Parameter | Meaning |
| --- | --- |
| `temperature` | Nonnegative temperature in K. |
| `start` | Optional `ForceConstants` containing FC2. The caller must use the same physical parameter layout and masses; this compatibility is not checked. Omitted means start from the model's bare FC2. |
| `mixing` | Linear mixing fraction in `(0, 1]` for each FC2 update. |
| `tolerance` | Convergence threshold for the star-weighted RMS frequency change, in THz. |
| `max_iterations` | Maximum self-consistency updates. |

Each iteration computes the FC4 loop correction from the current FC2, mixes the target update, and compares the resulting signed frequencies. `SCPHResult` contains `temperature`, the effective `fc2`, representative-point `frequencies`, `stars`, iteration `history`, `converged`, and `minimum_mode_thz`. `iterations` and `has_imaginary_modes` are convenience properties. Imaginary frequencies are reported as negative signed values and do not by themselves mean the iteration failed. The covariance calculation treats negative curvature by its magnitude; a non-translational exact zero mode has divergent harmonic covariance and raises an error.

Each `SCPHStep` records `index`, `frequency_change_thz`, and `minimum_frequency_thz`.

## Run a temperature series

```python
results = scph.run_many(
    [0, 100, 200, 300],
    mixing=0.2,
    tolerance=1e-9,
    max_iterations=200,
)
```

`run_many(temperatures, **kwargs)` requires a nonempty, strictly increasing sequence of nonnegative temperatures. Execution proceeds from highest to lowest temperature, while the returned `SCPHResult` list follows the input's ascending order. A converged higher-temperature FC2 initializes the next lower temperature; a nonconverged result does not seed the next run. Imaginary modes remain diagnostics, separate from numerical convergence.

The repository's [K4As4Pt2 SCPH script](https://github.com/gtiders/mlfcs/blob/dev/tutorial/K4As4Pt2/scph/run.py) shows a temperature series, high-to-low warm starts, and band plotting from the returned FC2 models.
