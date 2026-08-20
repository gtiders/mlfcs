# Native SSCHA module

[中文]

`mlfcs.anharmonic.sscha` iteratively fits an effective FC2 from ASE force snapshots and samples the next
harmonic canonical ensemble. It uses the native MLFCS symmetry-reduced Gram fitter and a compact
q-space sampler.

## Method

The first iteration uses small Gaussian Cartesian displacements unless an initial FC2 is supplied.
Every completed force batch is fitted with `ForceConstantFitter(orders=(2,))`, all snapshots are
used, and ASR is imposed in the irreducible parameter space. Subsequent structures are drawn from
the current harmonic Hamiltonian:

```text
initial Cartesian snapshots → ASE forces → native Gram FC2 fit
                                             │
                                             ▼
compact-FC2 q-space ensemble → ASE forces → native Gram FC2 fit → repeat
```

The sampler Fourier-transforms translation-reduced FC2 at reciprocal quotient q points defined by
the reference-supercell matrix. It diagonalizes `3 * primitive_atoms` matrices rather than one full
`3 * supercell_atoms` matrix. Conjugate q points are paired explicitly, and the three
mass-weighted translations are projected out at Gamma.

Quantum statistics are the default:

```text
variance(q_s) = hbar / (2 omega_s) coth[hbar omega_s / (2 kB T)].
```

Set `statistics="classical"` for `kB T / omega_s**2`.

## Direct ASE workflow

```python
from ase.io import read
from mlfcs.anharmonic.sscha import SSCHA

sscha = SSCHA(
    read("POSCAR"),
    reference=read("reference-supercell.vasp"),
    temperature=300.0,
    statistics="quantum",
    snapshots=1000,
    max_iterations=10,
    initial_displacement=0.01,
    random_seed=42,
    cutoff_frequency=0.01,  # THz
    imaginary_modes="error",
    max_displacement=None,  # canonical sampling is not clipped
    mixing=1.0,             # direct fixed-point update
    log_level=1,
)
sscha.run(make_my_ase_calculator())
```

`run()` evaluates one structure at a time. Use `calculate_free_energy=False` when energies and the
variational free-energy estimate are not required.

`temperature` may also be a sequence. It is sorted in ascending order and, by default, the final
FC2 at one temperature initializes the next calculation. Set `continuation=False` for independent
temperatures. SSCHA evaluates its internally generated ensemble through the supplied ASE calculator;
it intentionally does not expose finite-difference-style `sow()`/`reap()` interfaces.

## Stability controls

- `cutoff_frequency` excludes non-translational modes below the given absolute frequency in THz.
- `imaginary_modes="error"` is the default and rejects an unstable trial Hamiltonian.
- `imaginary_modes="absolute"` samples using the absolute frequency and records a warning-level
  diagnostic.
- `imaginary_modes="exclude"` removes imaginary modes from sampling.
- `max_displacement=None` preserves the canonical distribution. A positive value enables
  phonopy-style per-atom radial clipping: direction is retained and only vector length is shortened.
  The number of clipped atoms and affected snapshots is reported because clipping makes the sample
  distribution non-canonical.

`SSCHAIteration.ensemble` records q-point, mode, imaginary-mode, exclusion, and clipping counts.
`fitting_relative_force_error` records the native fitter's training error and
`relative_force_constant_change` records the update relative to the FC2 that generated the current
canonical ensemble after linear mixing. `raw_relative_force_constant_change` records the same
quantity before mixing. The initialization round has no relative FC2 change. These scalar diagnostics
keep the public iteration object compact; the internal sampling Hamiltonian is not duplicated in
the public API.

`mixing` controls the self-consistent update, not force-constant regression:

```text
Phi_next = (1 - mixing) Phi_sampled + mixing Phi_fitted.
```

The default `mixing=1` exactly reproduces direct replacement. Values below one under-relax the
sampling Hamiltonian used in the following iteration. This is useful when finite stochastic samples
make the fixed-point path oscillate; it does not alter the fitted IFC for the current sample.

## Results and output

```python
previous = sscha.force_constants
iteration = sscha.step(calculator)

sscha.write("FORCE_CONSTANTS", format="text")
sscha.write("fc2-300K.hdf5", format="hdf5")
```

`force_constants` uses full MLFCS internal supercell order. The active translation-reduced array is
available as `compact_force_constants`. History stores diagnostics only. Text and HDF5 output use the
shared phonopy-compatible MLFCS writers without importing phonopy.

## Free energy

The same commensurate q-point eigensolutions provide the quantum harmonic free energy per primitive
cell. The free energy of an iteration belongs to the trial FC2 that generated its snapshots, while
the active FC2 is the newly fitted FC2 for the next update. With snapshot
energies, the reported estimator is

```text
F = F_harm + mean[(E(u) - E(0) - 1/2 u Phi u) / number_of_cells].
```

The error is the standard error of the sampled correction. If `max_displacement` clips any sample,
the distribution is no longer strictly canonical and the estimator must be interpreted as an
approximation.
