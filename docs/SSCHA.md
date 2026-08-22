# SSCHA module

[中文](SSCHA_ZH.md)

`mlfcs.sscha` uses phonopy to sample thermal displacements, symfc to fit effective second-order
force constants, and any user-owned ASE Calculator for energies and forces. It is an isolated
optional module: the base order-parameterized finite-difference pipeline does not depend on
phonopy or symfc.

## Installation

```bash
uv sync --extra sscha
```

The optional dependencies are phonopy for supercells, harmonic canonical sampling, phonons, and
thermodynamic quantities, and symfc for symmetry-aware FC2 fitting and sum rules. Users install
and construct their actual calculator; MLFCS does not require calorine, MACE, GAP, or another
specific potential package.

## Iteration

Without supplied initial force constants, iteration zero fits FC2 from small random Cartesian
displacements. Each subsequent iteration samples the harmonic canonical ensemble defined by the
current FC2, evaluates the real potential on those configurations, and refits FC2:

```text
small Cartesian displacements → ASE forces → initial symfc FC2
                                                │
                                                ▼
canonical phonopy snapshots → ASE forces → refitted symfc FC2 → repeat
```

`max_iterations=10` therefore produces eleven fits including initialization. If
`initial_force_constants` is supplied, it is used for the first canonical sampling step instead
of Cartesian initialization. This stochastic effective-harmonic procedure is distinct from an
explicit FC3 bubble, FC4 loop, or deterministic ALAMODE SCPH expansion; anharmonicity enters
through the calculator forces on thermally displaced structures.

## Direct ASE workflow

```python
from ase.io import read
from mlfcs.sscha import SSCHA

atoms = read("POSCAR")
calculator = make_my_ase_calculator()

sscha = SSCHA(
    atoms,
    supercell=(3, 3, 3),
    temperature=300.0,
    snapshots=1000,
    max_iterations=10,
    initial_displacement=0.01,
    random_seed=42,
    symprec=1e-5,
    log_level=1,
)
sscha.run(calculator)
```

`run()` evaluates configurations serially to avoid replicating a large calculator. A progress
callback may be supplied. Energies are evaluated by default for the free-energy estimate; use
`calculate_free_energy=False` when only effective FC2 is required.

## External sow/reap workflow

```python
structures = sscha.sow()

for atoms in structures:
    iteration = atoms.info["mlfcs_sscha_iteration"]
    configuration_id = atoms.info["mlfcs_configuration_id"]
    dispatch(iteration, configuration_id, atoms)

result = sscha.reap(
    forces_by_configuration_id,
    energies=energies_by_configuration_id,
    reference_energy=equilibrium_supercell_energy,
)
```

Within one iteration, IDs must cover `0..N-1`. A positional array must follow `sow()` exactly; a
mapping may arrive in any insertion order but must contain every ID once. Force shape is
`(snapshots, supercell_atoms, 3)`. ASE units are used: Å, eV/Å, eV, and eV/Å².

Forces are the only required fitting input. Snapshot energies and the undisplaced-supercell
reference energy are used only for the free-energy estimate; if either is absent,
`free_energy` and `free_energy_error` are `None`.

## Results and convergence

Every `reap()` or `step()` returns an immutable `SSCHAIteration` and appends it to
`sscha.history`. It records the zero-based index, sampling mode, full-supercell FC2, free energy
per primitive cell, its finite-sample standard error, and average real and harmonic potential
energies.

The API performs exactly the requested number of iterations and does not impose a universal
stopping threshold. Call `step()` explicitly and monitor FC2, phonon frequencies, or free energy:

```python
import numpy as np

previous = sscha.force_constants
result = sscha.step(calculator)
if previous is not None:
    rms = np.sqrt(np.mean((result.force_constants - previous) ** 2))
    print("FC2 RMS change:", rms)
```

Random fits may fluctuate near convergence. Average the last iterations explicitly:

```python
average = sscha.averaged_force_constants(last=5)
sscha.use_average(last=5)
```

Write phonopy-native full FC2 and continue with the underlying Phonopy object:

```python
sscha.write("fc2-300K.hdf5", format="hdf5")
sscha.write("FORCE_CONSTANTS", format="text")

ph = sscha.phonopy
ph.run_mesh([20, 20, 20])
ph.run_thermal_properties(temperatures=[300])
```

## Free-energy estimate

For current FC2 $\Phi$, the implementation follows the phonopy-style estimator

```text
F = F_harm + mean[(E(u) - E(0) - 1/2 u Φ u) / n_cell].
```

The reported error is the standard error of the finite-snapshot mean in brackets. It does not
include FC2 fitting uncertainty or systematic error between self-consistent iterations.

## Stability and memory

- Canonical sampling requires a sufficiently stable current FC2. Supply stable initial force
  constants or perform a zero-temperature harmonic calculation for unstable starting points.
- `cutoff_frequency` removes very low-frequency modes and `max_displacement` limits amplitudes.
- A fixed `random_seed` reproduces base random samples, although Cartesian displacements still
  change when the FC2 eigenvectors change.
- Full FC2 storage scales as `N²×3×3`; snapshot forces and symfc bases can dominate memory.
- `run()` is serial by design. Use `sow()` / `reap()` for user-controlled external parallelism.
- Configure non-analytic corrections through `sscha.phonopy.nac_params` with phonopy-compatible
  Born charges and dielectric tensors.

Unlike the earlier `MLPSSCHA` interface, the current module accepts ASE calculators directly,
provides complete per-iteration sow/reap, exposes random seeds and structured history, reports a
finite-sample free-energy error, and makes averaging and file output explicit operations.
