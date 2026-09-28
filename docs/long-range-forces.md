# Long-range forces in polar crystals

In a polar crystal, dipole-dipole interactions produce a long-range harmonic force-constant component. A finite-range cluster model can represent the short-range part efficiently, while the long-range part is calculated from Born effective charges and the dielectric tensor. MLFCS exposes this split explicitly through the ASE calculator `Ewald`.

## The physical split

Write the harmonic force constants as

$$
\Phi^{\mathrm{total}} = \Phi^{\mathrm{short}} + \Phi^{\mathrm{Ewald}}.
$$

For a displacement $u$, the Ewald calculator returns harmonic forces $F^{\mathrm{Ewald}}=-\Phi^{\mathrm{Ewald}}u$. Given training forces $F^{\mathrm{data}}$, fit the short-range model to

$$
F^{\mathrm{short}} = F^{\mathrm{data}} - F^{\mathrm{Ewald}},
$$

then assemble the total FC2 tensor by adding the Ewald tensor to the fitted short-range FC2 tensor. The same mapping and the same displaced geometry must be used for both force terms.

## `Ewald` constructor and parameters

```python
from mlfcs.tools import Ewald

ewald = Ewald(
    space,
    mapping,
    born,
    dielectric,
    accuracy=1e-10,
    eta=None,
)
```

The full signature is `Ewald(space, mapping, born, dielectric, *, accuracy=1e-10, eta=None)`.

| Parameter | Meaning |
| --- | --- |
| `space` | `ClusterSpace` describing the primitive motif. |
| `mapping` | `ClusterMap` for the same cluster space and the fixed 3D periodic supercell. |
| `born` | Born effective-charge tensors, shape `(n_primitive, 3, 3)`, in elementary-charge units. The first index follows primitive site order; tensor indices are electric-field and displacement directions. The physical tensors should satisfy charge neutrality when summed over primitive sites. |
| `dielectric` | Electronic dielectric tensor, shape `(3, 3)`, dimensionless. The implementation symmetrizes its input and requires a positive-definite result. |
| `accuracy` | Ewald real-/reciprocal-sum target, strictly between 0 and 1; default `1e-10`. Smaller values request more complete lattice sums. |
| `eta` | Optional positive Ewald splitting parameter in Å$^{-1}$. `None` selects it from the transformed cell volume. Changing `eta` changes the real/reciprocal partition, not the converged physical sum. |

`space` and `mapping.space` must identify the same model. The calculator fixes the cell and atom sequence to `mapping.supercell`; structures passed to it must use that cell, periodicity, and atom order.

## Force and FC2 outputs

`Ewald` implements the ASE calculator properties `energy` and `forces`. For a displaced ASE structure with the mapped supercell:

```python
long_range_forces = ewald.get_forces(atoms)
long_range_fc2 = ewald.fc2
```

`get_forces(atoms)` evaluates the harmonic long-range force at that exact geometry. `ewald.fc2` returns a copy with shape `(n_primitive, n_supercell, 3, 3)`, primitive-first in the same compact convention as `ForceConstants.get(2, mapping)`, and units eV/Å². The Ewald FC2 construction sets the on-site block to the negative sum of the other blocks, so the finite-supercell long-range component satisfies ASR by construction.

## Fit the short-range forces and assemble total FC2

```python
from ase.calculators.singlepoint import SinglePointCalculator
from mlfcs import FitSystem
from mlfcs.force_constants import write_phonopy

short_frames = []
for frame in training_frames:
    short = frame.copy()
    short.calc = SinglePointCalculator(
        short,
        forces=frame.get_forces() - ewald.get_forces(frame),
    )
    short_frames.append(short)

system = FitSystem.from_atoms(mapping, short_frames)
short_model = system.force_constants(system.solve())
short_model = short_model.enforce_asr(orders=(2,)).force_constants

total_fc2 = short_model.get(2, mapping) + ewald.fc2
write_phonopy("fc2-total.hdf5", total_fc2, mapping, format="phonopy_hdf5")
```

The force subtraction is performed frame by frame, not as a post-fit correction to parameters. Projecting the short-range model onto ASR is consistent with the Ewald component already satisfying ASR; their sum then satisfies the same translational constraint. The array export shown here writes FC2 only. Model-based format support is documented under [force constants](force-constants.md).

## What this calculation represents

The Ewald tensor is a finite-supercell harmonic dipole correction. It does not itself add the non-analytic $q\to0$ LO-TO splitting used for polar phonon bands. The exported FC2 file therefore does not replace the separate Born-charge/dielectric input required by a downstream phonon code's NAC feature.

The [NaCl tutorial](../tutorial/NaCl/README.md) contains the complete direct-fit versus long-range-subtracted workflow and its phonon-band comparison. Its `BORN` file supplies the Born and dielectric tensors; [the fit script](../tutorial/NaCl/fit.py) produces `fc2-direct.hdf5` and `fc2-corrected.hdf5`.
