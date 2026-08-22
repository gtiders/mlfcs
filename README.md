# MLFCS

English | [中文](README_ZH.md)

MLFCS is an ASE-first Python library for calculating symmetry-reduced force constants from
atomic forces. It supports second and arbitrarily higher orders through one order-parameterized
pipeline, with third- and fourth-order calculations as the primary production-validated paths.
Fifth and higher orders can also be calculated and exported through the generic sparse HDF5
format; practical size is determined by cluster count, cutoff, supercell size, and available
memory.

Finite differences use ASE, NumPy, and SciPy only, so an external calculator retains complete
control over CPU or GPU execution. The joint fitter supports CPU and GPU: a CUDA-enabled JAX
installation accelerates only the dense Wick feature kernel, while geometry, symmetry, sparse
constraints, and final solving remain host-side. Memory is controlled through symmetry reduction,
contiguous sparse arrays, lazy dense materialization, constraint null-space coordinates, and
bounded feature tiles. Static JAX buffers and compiled kernels are prepared once per fit and reused
for training, validation, and diagnostics.

The base package does not prescribe how forces are generated. Structures can be evaluated with
any user-owned ASE Calculator or dispatched to an external workflow. The native SSCHA module
combines q-space quantum harmonic sampling with the MLFCS Gram fitter to calculate
temperature-dependent effective second-order force constants.

MLFCS provides a Python API only; it has no CLI.

> **Development branch:** The joint force-only FC2--FCn fitting API described below is currently
> developed on the `dev` branch. The stable `main` branch retains the finite-difference API and
> common force-constant I/O. This label will be removed when fitting validation is promoted.

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
sparse symmetry reconstruction and optional sum-rule projection
        │
        ▼
ForceConstants → HDF5 / ShengBTE / phonopy-compatible formats
```

Space-group symmetry, force-constant index permutations, and stabilizer constraints reduce each
cluster tensor to independent components. Only those components are sampled. The reconstructed
result remains sparse until a dense representation is explicitly requested.

The acoustic sum rule (ASR) is imposed as a constrained projection in the independent orbit
parameter space:

```text
sum over one atom index of Phi(i1, ..., in) = 0
```

All constraint systems use a sparse, matrix-free LSMR projection. Harmonic calculations may also
enable the optional Born-Huang rotational sum rules; see [Sum rules](docs/en/methods/sum-rules.md).

## Features

- One API and one numerical pipeline for `order >= 2`.
- Production-tested third- and fourth-order force constants.
- Analytic FC4 validation against an independently differentiated FCC Morse energy, including
  second-order finite-difference step convergence.
- End-to-end second- and fifth-order validation.
- ASE `Atoms` and ASE `Calculator` at the public boundary.
- External, checkpoint-friendly `sow()` / `reap()` workflow.
- Optional direct-calculator zero-step extrapolation with configurable even-power degree.
- Stable configuration IDs and explicit atom-order mappings.
- Joint periodic-image cluster cutoff geometry.
- Recursive central-difference stencils.
- CPU/GPU fitting with persistent JAX Wick-feature kernels; finite differences stay host-side.
- Reusable JAX JIT and batched contractions for high-rank fitting throughput.
- Contiguous sparse arrays, matrix-free actions, and lazy materialization to reduce peak memory.
- Displacement-key deduplication to reduce expensive calculator evaluations.
- Joint force-only FC2--FCn fitting with Wick-orthogonalized features and Taylor-compatible
  force-constant output.
- Strict translational ASR using sparse matrix-free LSMR.
- Optional Born-Huang rotational sum rules for FC2.
- Generic sparse HDF5 for any order.
- ShengBTE output for orders 3 and 4.
- Full dense phonopy text output for order 2.
- Native quantum/classical SSCHA with arbitrary ASE calculators.

## Installation

MLFCS requires Python 3.12 or newer. Install and run it with uv:

```bash
uv sync
```

Runnable API examples are available in [`examples/`](examples/):

- [`basic_fc2.py`](examples/basic_fc2.py) runs FC2 directly with ASE's built-in EMT calculator;
- [`vasp_external_fc3.py`](examples/vasp_external_fc3.py) implements a complete external VASP
  `sow` / force collection / `reap` workflow;
- [`nep89_orders.py`](examples/nep89_orders.py) evaluates one or more orders with a user-supplied
  NEP89 model through calorine's ASE calculator.

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
| `None` cutoff | Maximum radius enumerable by the supercell |

JAX numerical kernels use 64-bit floating point.

## Choose the downstream structure first

Before calculating force constants, decide which program will consume them and establish its
primitive-cell and supercell conventions first. For the most reliable workflow, use the primitive
cell and reference supercell generated or validated by that downstream program, then pass those
same ASE `Atoms` objects to MLFCS. MLFCS can validate and convert equivalent primitive and
supercell representations, including atom reorderings and integral basis changes, but starting
from the downstream program's own structures avoids unnecessary mapping ambiguity at the final
export boundary.

## Quick start

### External force workflow

`sow()` does not write files or run a DFT program. It returns an ordered Python list of displaced
ASE `Atoms` objects. The user chooses how to serialize and calculate them: ASE can write each
structure as `POSCAR-xxx` for VASP or in another calculator's input format, after which the jobs
can be submitted by any local scheduler. When the calculations finish, use ASE to read each
result, extract its forces, restore the sow order (or key the forces by configuration ID), and
pass only those forces to `reap()`. If this positional order is guaranteed, configuration IDs
are not required.

For example, a positional VASP workflow is:

```python
from pathlib import Path

import numpy as np
from ase.io import read, write
from mlfcs import ForceConstantCalculation

calculation = ForceConstantCalculation(
    read("POSCAR"),
    order=3,
    supercell=(2, 2, 2),
    cutoff=-5,
    displacement=0.01,
    symprec=1e-5,
)

# 1. sow(): obtain the displaced ASE structures in the exact reap order.
structures = calculation.sow()
Path("vasp-jobs").mkdir(exist_ok=True)
for configuration_id, atoms in enumerate(structures):
    job = Path("vasp-jobs") / f"POSCAR-{configuration_id + 1:03d}"
    job.mkdir(exist_ok=True)
    write(job / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)

# 2. The user supplies INCAR, KPOINTS, POTCAR and submits every directory.
#    MLFCS does not launch or configure VASP.

# 3. Read completed results with ASE in the same filename/order convention.
forces = []
for configuration_id in range(len(structures)):
    job = Path("vasp-jobs") / f"POSCAR-{configuration_id + 1:03d}"
    completed = read(job / "vasprun.xml", index=-1)
    forces.append(completed.get_forces())
forces = np.asarray(forces)

# 4. reap(): the force at forces[i] must belong to structures[i].
fc3 = calculation.reap(forces, acoustic_sum_rule=True)
fc3.write("fc3.h5", format="hdf5")
```

For Quantum ESPRESSO, ABINIT, CP2K, or another external program, replace only the ASE `write()`
and `read()` formats and provide that program's required input parameters. The sow/reap contract
is unchanged. Positional `reap()` needs no metadata when file names and returned forces preserve
the exact sow order. File formats such as POSCAR do not preserve the Python `atoms.info` metadata,
so a manifest containing the filename-to-configuration-ID relation is recommended
for out-of-order jobs, restarts, long-term archives, and accidental-dataset detection. The complete
[`vasp_external_fc3.py`](examples/vasp_external_fc3.py) example implements this optional safety
layer, force collection, missing-result checks, and final export; see the
[external VASP workflow guide](docs/en/workflows/external-calculators.md).

The force array must have shape:

```text
(len(calculation.sow()), len(calculation.supercell), 3)
```

While the structures remain as ASE objects, every displaced structure also contains optional
audit metadata:

```python
atoms.info["mlfcs_configuration_id"]
atoms.info["mlfcs_atom_order"]
atoms.arrays["mlfcs_displacement"]
```

When jobs return out of order, read them in any order and pass a mapping keyed by the original
zero-based configuration ID:

```python
fc3 = calculation.reap(forces_by_configuration_id)
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

Direct ASE-calculator runs can optionally extrapolate the selected order to zero displacement:

```python
fc4 = calculation.run(
    calculator,
    derivative_backend="extrapolate",
    extrapolation_spacing=0.005,
    extrapolation_side_steps=2,
    extrapolation_degree=1,
)
```

For a central displacement of `0.03` Å, this samples `0.02`, `0.025`, `0.03`, `0.035`, and
`0.04` Å. The default degree `1` fits `D(h) = D0 + c2 h²`; higher degrees fit additional even
powers. This backend is intentionally available only through `run()`, not external `sow()` /
`reap()`. See [Zero-step extrapolation](docs/en/workflows/extrapolation.md).

For explicit checkpointing:

```python
forces = calculation.evaluate(calculator)
np.savez_compressed("forces.npz", forces=forces)
fc3 = calculation.reap(forces)
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
full_calculation = ForceConstantCalculation(atoms, order=3, cutoff=None)
```

A positive cutoff is a radius in Å. A negative integer selects a one-based neighbor shell. For
example, `cutoff=-8` selects the eighth shell. `cutoff=None` selects the maximum radius that the
current finite supercell can enumerate; it does not mean an unbounded interaction range. MLFCS
reports both the supercell capacity and the selected radius:

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

## Atom ordering and structure frames

The reference supercell order is authoritative. `sow()` returns that exact order, and every force
array passed to `reap()` must use it unchanged. There is no internal/grouped atom-order mode and
MLFCS never silently reorders external forces or fitting snapshots. Construct a calculation either
from an integer general supercell matrix or from an explicit reference frame:

```python
calculation = ForceConstantCalculation(
    primitive, order=3, supercell_matrix=[[2, 1, 0], [0, 2, 0], [0, 0, 1]]
)
# or
calculation = ForceConstantCalculation(primitive, reference=reference_supercell, order=3)
```

To build that reference explicitly, use the stable public helper:

```python
from mlfcs import build_supercell

reference_supercell = build_supercell(primitive, [[2, 1, 0], [0, 2, 0], [0, 0, 1]])
```

Format writers create any required format-specific ordering only at the export boundary.
For independently reordered snapshots, call `mlfcs.align_structures(reference, atoms)` explicitly;
it returns the aligned structure and its maximum periodic matching residual.

## Acoustic sum rule

ASR is enabled by default:

```python
constrained = calculation.reap(forces, acoustic_sum_rule=True)
raw = calculation.reap(forces, acoustic_sum_rule=False)
```

The constrained result is the nearest solution in independent parameter space that satisfies
translational invariance. Permutation symmetry supplies equivalent constraints on the other atom
axes.

Born-Huang and Huang conditions are explicit FC2-only postprocessing, shared by finite
difference and fitting results. The strict default is `strength=1.0`; FC3 and higher orders are
not changed:

```python
constrained = result.enforce_harmonic_constraints(
    born_huang=True,
    huang=True,
)
```

The projector always enforces FC2 ASR, uses all tied nearest periodic images, and reports its
residuals and correction. `strength` in `[0, 1]` scales only the Born-Huang/Huang correction.
See [Sum rules](docs/en/methods/sum-rules.md).

## Output formats

The output format is always explicit:

```python
fc2.write("FORCE_CONSTANTS", format="phonopy")
fc2.write("fc2.hdf5", format="phonopy_hdf5")
fc3.write("fc3.h5", format="hdf5")
fc3.write("fc3.hdf5", format="phono3py_hdf5")
fc3.write("FORCE_CONSTANTS_3RD", format="shengbte")
fc4.write("FORCE_CONSTANTS_4TH", format="shengbte")
fc234.write("force_constants.xml", format="alamode")
```

Read the native sparse HDF5 format through the matching public API:

```python
from mlfcs import read_hdf5

fc234 = read_hdf5("fc3.h5")
```

| Format | Orders | Representation |
|---|---|---|
| `hdf5` | Any | Native v2 lattice-labelled sparse IFCs (`sites`, translation representatives, Cartesian tensors) |
| `shengbte` | 3 and 4 | Symmetry-closed translation-based text blocks |
| `phonopy` | 2 | Full dense supercell FC2 text |
| `phonopy_hdf5` | 2 | Phonopy-compatible full-supercell `force_constants` HDF5 |
| `phono3py_hdf5` | 3 | Phono3py-compatible full-supercell `fc3` HDF5 |
| `alamode` | 2--4 | Combined ALAMODE FCSXML document |

ShengBTE output writes the symmetry-closed cluster support carried by the reconstructed sparse
result and resolves its lattice residues to jointly compatible minimum images.

The phonopy and phono3py HDF5 writers preserve the explicit reference-supercell order and stream
one first-atom slab at a time. They therefore do not materialize the full FC3 in memory. The native
`hdf5` format is native schema v2: it stores primitive and reference structures, their verified
mapping, and lattice-labelled sparse IFCs. Older native schemas are intentionally unsupported.

ALAMODE XML preserves the exact atom order of `fc.supercell`. Primitive-atom identities and
translation mappings come exclusively from MLFCS's `primitive_index` and `cell_translation`
metadata; export does not ask spglib or ALAMODE to rediscover or reorder the cell. Use `order=2`,
`3`, or `4` to write one available order, or omit it to combine all available FC2--FC4 orders.
See the [ALAMODE XML guide](docs/en/formats/alamode.md) for the mapping and periodic-image contract.

Sparse HDF5 is recommended for high orders. Dense materialization is explicit and emits a
warning above the default 2 GB advisory budget:

```python
fc5.write("fc5.h5", format="hdf5")
dense = fc5.materialize(5)
dense = fc5.materialize(5, max_bytes=None)  # explicitly disable the warning budget
```

## Native SSCHA

The independent `mlfcs.sscha` module fits a temperature-dependent effective harmonic FC2 from
thermally sampled forces:

```python
from mlfcs.sscha import SSCHA

sscha = SSCHA(
    atoms,
    supercell=(3, 3, 3),
    temperature=300,
    statistics="quantum",
    snapshots=1000,
    max_iterations=10,
    random_seed=42,
)

sscha.run(calculator)
sscha.use_average(last=5)
sscha.write("fc2-300K.hdf5", format="hdf5")
```

Iteration zero fits an initial FC2 from small random Cartesian displacements. Each subsequent
iteration samples the canonical harmonic ensemble in compact-FC2 q space and refits FC2 with the
native streamed-Gram fitter. Thus `max_iterations=10` performs one initialization and ten updates.

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
Completed iterations and sampling diagnostics are stored in `sscha.history`. Phonopy-compatible
text and HDF5 output use the shared MLFCS writers without requiring phonopy at runtime.
Canonical iterations also report the relative FC2 update, while the trial sampling Hamiltonian
remains an internal detail.

This is a stochastic effective-harmonic method, not an explicit FC3 bubble or FC4 loop
calculation. See the [SSCHA guide](docs/en/workflows/sscha.md) for details.

## Current limitations

- Third and fourth order are the main production-tested finite-difference paths.
- Higher orders use the same implementation but can become prohibitively expensive.
- ShengBTE output is limited to orders 3 and 4.
- Explicit FC3 bubble and FC4 loop self-energies are not implemented.
- Non-analytic electrostatic corrections are not part of the base reconstruction pipeline.
- SSCHA stopping criteria are controlled by the caller; no universal automatic convergence
  threshold is imposed.

## Documentation

- Full bilingual docs: [English site](https://gtiders.github.io/mlfcs/) and [中文 site](https://gtiders.github.io/mlfcs/zh/). See [runnable examples](examples/README.md) and [development validation](docs/en/development/validation.md)

## Development

All commands use uv and tests run serially:

```bash
uv sync
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv build
```

Local tests are deterministic unit and public-API regressions. Material comparisons and third-party
transport workflows are documented under `examples/cases/` and are run manually. CI only builds
the bilingual documentation sites. The test organization is documented in [tests/README.md](tests/README.md).

The current development version is `4.0.0a2` (4.0 alpha 2). See [CHANGELOG.md](CHANGELOG.md) for release notes and
[CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## License

MLFCS is distributed under the [GNU General Public License v3.0 or later](LICENSE). Adapted
third-party components and redistributed reference data are documented in
Third-party terms for the ALAMODE adapter are retained directly in its source module.
