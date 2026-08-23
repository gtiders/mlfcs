# MLFCS

[![Documentation](https://github.com/gtiders/mlfcs/actions/workflows/ci.yml/badge.svg)](https://github.com/gtiders/mlfcs/actions/workflows/ci.yml)
[![Documentation site](https://img.shields.io/badge/docs-GitHub%20Pages-0f766e)](https://gtiders.github.io/mlfcs/en/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776ab)](https://www.python.org/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

English | [简体中文](README.zh-CN.md)

<!-- BEGIN GENERATED: docs/en/index.md -->

<p align="center">
  <img src="https://gtiders.github.io/mlfcs/assets/images/logo.png" alt="MLFCS" width="560">
</p>



MLFCS is an ASE-first Python library for constructing symmetry-reduced harmonic and anharmonic force constants from atomic forces. It provides finite differences, force-only fitting, physical constraints, sparse primitive real-space storage, temperature-dependent effective harmonic workflows, and explicit export to downstream phonon and transport software.

## Why MLFCS

High-order force constants combine rapidly growing interaction spaces, Cartesian tensor symmetry, large training sets, and errors coupled across Taylor orders. MLFCS keeps structures, symmetry-reduced interactions, polynomial bases, fitting, force constants, and format conversion as explicit stages so that each approximation can be inspected and validated.

## Core capabilities

- FC2 and arbitrarily higher-order finite differences with production validation centered on FC2–FC4.
- Joint force-only fitting on one fixed reference supercell using Taylor-compatible Wick coordinates.
- Primitive-site plus exact integer-translation force constants with sparse native HDF5 v3 storage.
- Translational constraints and explicit FC2 Born–Huang/Huang post-processing.
- Target-supercell realization and writers for phonopy, phono3py, ShengBTE, and ALAMODE.
- FC4 loop SCPH and stochastic effective-harmonic SSCHA workflows.

MLFCS is a Python library, not a command-line application. Force generation remains under the user's ASE calculator or external electronic-structure workflow.

## Quick start

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from mlfcs import ForceConstantCalculation, build_supercell, write_force_constants

primitive = bulk("Al", "fcc", a=4.05)
reference = build_supercell(primitive, (2, 2, 2))
calculation = ForceConstantCalculation(
    primitive,
    reference=reference,
    order=2,
    cutoff=None,
)
fc2 = calculation.run(EMT())
write_force_constants(fc2, "mlfcs.h5", format="hdf5")
```

See the [first finite-difference FC2 tutorial](docs/en/tutorials/first-fc2-finite-difference.md) before applying the workflow to an external calculator or a higher order.

## Typical workflows

| Goal | Start here |
|---|---|
| Calculate force constants from an ASE calculator | [Finite-difference workflow](docs/en/tutorials/finite-difference-workflow.md) |
| Send structures to VASP or another external program | [External calculator tutorial](docs/en/tutorials/external-calculator.md) |
| Fit force constants from displaced or MD snapshots | [First fitting tutorial](docs/en/tutorials/first-fc2-fitting.md) |
| Fit FC2, FC3, and FC4 together | [Joint high-order fitting](docs/en/tutorials/joint-fc2-fc3-fc4.md) |
| Apply FC2 rotational conditions | [Rotational constraints](docs/en/tutorials/rotational-constraints.md) |
| Calculate an FC4 loop correction | [SCPH workflow](docs/en/tutorials/scph-workflow.md) |
| Calculate a stochastic effective FC2 | [SSCHA workflow](docs/en/tutorials/sscha-workflow.md) |
| Export to another program or supercell | Interoperability |

## Choose structures before calculation

Decide which downstream program will consume the result before generating force constants. Prefer its primitive and reference supercell, keep the reference atom order unchanged through force collection, and use target realization only for validated integer-supercell representations. MLFCS does not silently redefine a primitive cell, enlarge a training supercell, apply strain, or perform a rigid Cartesian rotation.

## Current scope and limitations

- One fit uses one fixed reference supercell; multi-supercell joint fitting is not supported.
- Third and fourth order are the main production-validated high-order paths; higher orders may be prohibitively expensive.
- Native HDF5 uses schema v3; older native schemas are intentionally rejected.
- ShengBTE output supports FC3 and FC4; ALAMODE output supports the implemented FC2–FC4 mapping.
- Long-range electrostatic force subtraction, multipole corrections, and explicit FC3 bubble self-energies are not implemented.
- SCPH and SSCHA require explicit convergence inspection.

The [capability status](docs/en/overview/capabilities.md) and [roadmap](docs/en/roadmap/index.md) distinguish stable, experimental, planned, research, and No-Go work.

## Documentation map

### Theory

Start with the [theory](docs/en/theory/index.md) for derivations and numerical conventions.

### Concepts

Use the core concepts to understand structures, translations, interactions, orbits, parameters, bases, and realizations.

### Tutorials

Follow the [tutorial learning paths](docs/en/tutorials/index.md) for complete, executable workflows.

### How-to guides

Use the task guides to complete focused tasks.

### Interoperability

Read formats and structure conventions before exporting.

### Examples

The material cases connect scripts, reference data, results, and figures.

### Q&A

The questions and answers route common problems to authoritative pages.

### API reference

The [manually maintained API reference](docs/en/reference/index.md) records public contracts.

### Roadmap

The [roadmap](docs/en/roadmap/index.md) separates stable, planned, research, and No-Go work.

### Developer documentation

The developer guide covers architecture, testing, validation, and maintenance.

<!-- END GENERATED: docs/en/index.md -->

## Documentation

Read the [English documentation](https://gtiders.github.io/mlfcs/en/) or the [Chinese documentation](https://gtiders.github.io/mlfcs/zh/). The source pages live under [docs/en](docs/en/) and [docs/zh](docs/zh/).

## Citation

If MLFCS contributes to published work, cite the software metadata in [CITATION.cff](CITATION.cff).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), the [issue tracker](https://github.com/gtiders/mlfcs/issues), and the [developer documentation](https://gtiders.github.io/mlfcs/en/development/).

## License

MLFCS is distributed under the [GNU General Public License v3.0 or later](LICENSE).
