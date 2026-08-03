# MLFCS

English | [中文](README_ZH.md)

MLFCS is an ASE-first Python library for calculating symmetry-reduced force constants from
atomic forces. It supports second and arbitrarily higher orders through one order-parameterized
pipeline, with third- and fourth-order calculations as the primary production-validated paths.
Fifth and higher orders can also be calculated and exported through the generic sparse HDF5
format; practical size is determined by cluster count, cutoff, supercell size, and available
memory.

Numerical execution supports both CPU and GPU. CPU mode handles ordinary calculations and
large sparse linear algebra, while a CUDA-enabled JAX installation can move high-rank Cartesian
tensor rotations and batched transformations to a GPU. Memory is controlled through symmetry
reduction, contiguous sparse arrays, lazy dense materialization, matrix-free tensor actions,
small Gram null spaces, and sparse LSMR. JAX JIT, `vmap`, batched contractions, and displacement
deduplication improve throughput. Actual gains depend on the system, order, and hardware; GPU
execution does not replace cluster enumeration or sparse solvers that still run on the CPU.

The base package does not prescribe how forces are generated. Structures can be evaluated with
any user-owned ASE Calculator or dispatched to an external workflow. An independent optional
module uses phonopy and symfc to calculate temperature-dependent effective second-order force
constants with a stochastic self-consistent harmonic approximation (SSCHA).

MLFCS provides a Python API only; it has no CLI.

## How it works

An order-`n` force constant is the `n`th derivative of potential energy with respect to atomic
displacements, or equivalently an `(n-1)`-fold derivative of force. MLFCS computes it as follows:

```text
ASE primitive structure
        │
        ▼
deterministic supercell and neighbor cutoff
        │
        ▼
space-group and permutation-reduced cluster orbits
        │
        ▼
recursive central-difference displacement plan
        │
        ▼
user-provided forces
        │
        ▼
sparse symmetry reconstruction and optional strict ASR
        │
        ▼
ForceConstants → HDF5 / NumPy / ShengBTE / phonopy
```

Space-group symmetry, force-constant index permutations, and stabilizer constraints reduce each
cluster tensor to independent components. Only those components are sampled. The reconstructed
result remains sparse until a dense representation is explicitly requested.

The acoustic sum rule (ASR) is imposed as a constrained projection in the independent orbit
parameter space:

```text
sum over one atom index of Phi(i1, ..., in) = 0
```

Small constraint systems use a Gram-matrix null space followed by sparse LSMR refinement; large
systems use a sparse LSMR projection directly.

## Features

- One API and one numerical pipeline for `order >= 2`.
- Production-tested third- and fourth-order force constants.
- End-to-end second- and fifth-order validation.
- ASE `Atoms` and ASE `Calculator` at the public boundary.
- External, checkpoint-friendly `sow()` / `reap()` workflow.
- Stable configuration IDs, plan hashes, and explicit atom-order mappings.
- Joint periodic-image cluster cutoff geometry.
- Recursive central-difference stencils.
- JAX-accelerated high-rank tensor transformations with CPU/GPU selection.
- JAX JIT, `vmap`, and batched contractions for high-rank tensor throughput.
- Contiguous sparse arrays, matrix-free actions, and lazy materialization to reduce peak memory.
- Displacement-key deduplication to reduce expensive calculator evaluations.
- Strict translational ASR using Gram null spaces and sparse LSMR.
- Generic sparse HDF5 for any order.
- ShengBTE output for orders 3 and 4.
- Full dense phonopy text output for order 2.
- Optional phonopy/symfc SSCHA with arbitrary ASE calculators.

## Installation

MLFCS requires Python 3.12 or newer. Install and run it with uv:

```bash
uv sync
```

Install the optional SSCHA dependencies when needed:

```bash
uv sync --extra sscha
```

Calculator packages such as calorine or MACE are intentionally not base dependencies. Install
the calculator required by your application separately.

## Units

| Quantity | Unit |
|---|---|
| Cell, positions, and displacements | Å |
| Forces | eV/Å |
| Order-`n` force constants | eV/Åⁿ |
| Positive cutoff | Å |
| Negative integer cutoff | Neighbor shell |

JAX numerical kernels use 64-bit floating point.

## Quick start

### External force workflow

```python
import numpy as np
from ase.io import read
from mlfcs import ForceConstantCalculation

calculation = ForceConstantCalculation(
    read("POSCAR"),
    order=3,
    supercell=(2, 2, 2),
    cutoff=-5,
    displacement=0.01,
    symprec=1e-5,
    jax_platform="auto",  # "auto", "cpu", or "gpu"
)

structures = calculation.sow()
forces = np.asarray(evaluate_structures(structures))

fc3 = calculation.reap(
    forces,
    plan_hash=calculation.plan.hash,
    acoustic_sum_rule=True,
)
fc3.write("fc3.h5", format="hdf5")
```

The force array must have shape:

```text
(len(calculation.sow()), len(calculation.supercell), 3)
```

Every displaced structure contains:

```python
atoms.info["mlfcs_configuration_id"]
atoms.info["mlfcs_plan_hash"]
atoms.info["mlfcs_atom_order"]
atoms.arrays["mlfcs_displacement"]
```

When jobs return out of order, pass a mapping keyed by configuration ID:

```python
fc3 = calculation.reap(
    forces_by_configuration_id,
    plan_hash=calculation.plan.hash,
)
```

Missing or extra IDs, invalid shapes, non-finite values, and plan-hash mismatches are rejected.

### Direct ASE Calculator workflow

```python
calculator = make_my_ase_calculator()

fc3 = calculation.run(
    calculator,
    progress=lambda done, total: print(f"{done}/{total}"),
)
```

Calculator evaluation is serial by design to avoid multiplying the memory used by large machine
learning potentials. Use `sow()` / `reap()` when external parallelism or checkpointing is needed.

For explicit checkpointing:

```python
forces = calculation.evaluate(calculator)
np.savez_compressed("forces.npz", forces=forces, plan_hash=calculation.plan.hash)
fc3 = calculation.reap(forces, plan_hash=calculation.plan.hash)
```

Stage reporting is enabled by default for `sow()`, `reap()`, and direct ASE-calculator runs.
It reports symmetry, cluster, displacement-plan, ASR, and force-evaluation progress using values
already computed by the calculation. Pass `verbose=False` to silence all stage and cutoff output.
`report_cutoff=False` suppresses only the detailed neighbor-shell lines.

## Orders and cutoffs

The same constructor is used for all supported orders:

```python
fc2_calculation = ForceConstantCalculation(atoms, order=2, cutoff=-6)
fc4_calculation = ForceConstantCalculation(atoms, order=4, cutoff=-3)
fc5_calculation = ForceConstantCalculation(atoms, order=5, cutoff=-1)
```

A positive cutoff is a radius in Å. A negative integer selects a one-based neighbor shell. For
example, `cutoff=-8` selects the eighth shell. MLFCS reports both the supercell capacity and the
selected radius:

```text
Supercell neighbor limit: maximum shell = 33, maximum cutoff radius = 15.7504983443 Å
Selected neighbor cutoff: shell = 8, cutoff radius = 7.5419604204 Å
```

The first line is a capacity diagnostic for the finite supercell. The second line is the cutoff
actually used. Requests beyond the enumerable capacity are rejected. Use `report_cutoff=False`
to suppress both lines.

Higher orders grow combinatorially through cluster combinations, tensor components,
permutations, and finite-difference signs. Use small cutoffs first and monitor configuration
count and memory.

## Atom ordering

The canonical internal supercell order is:

```text
z → y → x → primitive_atom
```

The primitive-atom index changes fastest. This is the default order used by `sow()` and `reap()`.

For primitive-atom-grouped data:

```python
structures = calculation.sow(atom_order="grouped")
force_constants = calculation.reap(forces, atom_order="grouped")
```

Explicit mappings are available as:

```python
calculation.index.grouped_permutation
calculation.index.internal_from_grouped
calculation.index.group_atoms(atoms)
```

MLFCS performs the required grouped-order conversion automatically at the phonopy output
boundary.

## Acoustic sum rule

ASR is enabled by default:

```python
constrained = calculation.reap(forces, acoustic_sum_rule=True)
raw = calculation.reap(forces, acoustic_sum_rule=False)
```

The constrained result is the nearest solution in independent parameter space that satisfies
translational invariance. Permutation symmetry supplies equivalent constraints on the other atom
axes.

## Output formats

The output format is always explicit:

```python
fc2.write("FORCE_CONSTANTS", format="phonopy")
fc2.write("fc2.hdf5", format="phonopy_hdf5")
fc3.write("fc3.h5", format="hdf5")
fc3.write("fc3.hdf5", format="phono3py_hdf5")
fc3.write("fc3.npz", format="numpy")
fc3.write("FORCE_CONSTANTS_3RD", format="shengbte")
fc4.write("FORCE_CONSTANTS_4TH", format="shengbte")
```

| Format | Orders | Representation |
|---|---|---|
| `hdf5` | Any | Sparse cluster tensors or dense arrays |
| `numpy` / `npz` | Any | Materialized NumPy arrays |
| `shengbte` | 3 and 4 | Symmetry-closed translation-based text blocks |
| `phonopy` | 2 | Full dense supercell FC2 text |
| `phonopy_hdf5` | 2 | Phonopy-compatible full-supercell `force_constants` HDF5 |
| `phono3py_hdf5` | 3 | Phono3py-compatible full-supercell `fc3` HDF5 |

ShengBTE output is faithful by default: it writes exactly the symmetry-closed cluster support
carried by the reconstructed sparse result. To reproduce the legacy thirdorder secondary
joint-image filtering and block order, request compatibility explicitly:

```python
fc3.write(
    "FORCE_CONSTANTS_3RD",
    format="shengbte",
    compatibility="thirdorder",
)
```

The phonopy and phono3py HDF5 writers use primitive-atom-grouped supercell order and stream one
first-atom slab at a time. They therefore do not materialize the full FC3 in memory. The native
`hdf5` format remains the compact, order-parameterized MLFCS representation.

Sparse HDF5 is recommended for high orders. Dense materialization is explicit and emits a
warning above the default 2 GB advisory budget:

```python
fc5.write("fc5.h5", format="hdf5")
dense = fc5.materialize(5)
dense = fc5.materialize(5, max_bytes=None)  # explicitly disable the warning budget
```

## Optional SSCHA

The independent `mlfcs.sscha` module fits a temperature-dependent effective harmonic FC2 from
thermally sampled forces:

```python
from mlfcs.sscha import SSCHA

sscha = SSCHA(
    atoms,
    supercell=(3, 3, 3),
    temperature=300,
    snapshots=1000,
    max_iterations=10,
    random_seed=42,
)

sscha.run(calculator)
sscha.use_average(last=5)
sscha.write("fc2-300K.hdf5", format="hdf5")
```

Iteration zero fits an initial FC2 from small random Cartesian displacements. Each subsequent
iteration uses phonopy to sample the canonical harmonic ensemble and symfc to refit full FC2.
Thus `max_iterations=10` performs one initialization and ten updates.

External per-iteration execution is also supported:

```python
structures = sscha.sow()
result = sscha.reap(
    forces_by_configuration_id,
    energies=energies_by_configuration_id,
    reference_energy=equilibrium_supercell_energy,
)
```

Forces are sufficient for FC2 fitting. Energies are required only for the free-energy estimate.
Completed iterations are stored in `sscha.history`, and `sscha.phonopy` exposes the underlying
Phonopy object for meshes, bands, DOS, and thermal properties.

This is a stochastic effective-harmonic method, not an explicit FC3 bubble or FC4 loop
calculation. See the [SSCHA guide](docs/SSCHA.md) for details.

## Current limitations

- Third and fourth order are the main production-tested finite-difference paths.
- Higher orders use the same implementation but can become prohibitively expensive.
- ShengBTE output is limited to orders 3 and 4.
- Explicit FC3 bubble and FC4 loop self-energies are not implemented.
- Non-analytic electrostatic corrections are not part of the base reconstruction pipeline.
- SSCHA stopping criteria are controlled by the caller; no universal automatic convergence
  threshold is imposed.

## Documentation

- [Technical overview](docs/TECHNICAL_OVERVIEW.md)
- [Numerical validation and CI (Chinese)](docs/VALIDATION_ZH.md)
- [SSCHA guide](docs/SSCHA.md)
- [Detailed old/new implementation comparison (Chinese)](docs/OLD_NEW_COMPARISON_ZH.md)

Implementation comparisons, compatibility decisions, benchmark counts, and measured memory
figures are intentionally kept in the comparison and technical documents rather than this user
introduction.

## Development

All commands use uv and tests run serially:

```bash
uv sync --extra sscha
uv run pytest
uv run pytest -m "not reference"
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv build
```

hiphive and phono3py are development-only validation dependencies. The CI reference compares
AlN FC3 values against an independent phono3py finite-difference result after hiphive converts
both atom orderings and tensor representations to the same full-supercell form.
The test hierarchy and independent reference commands are documented in
[tests/README.md](tests/README.md).

The current development version is `0.5.0`.

## License

MLFCS is distributed under the [Apache License 2.0](LICENSE).
