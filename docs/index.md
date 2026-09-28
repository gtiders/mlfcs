# MLFCS

**MLFCS** builds higher-order force-constant models from crystal structures and atomic forces.

Starting from a structural model and a user-supplied supercell, MLFCS can construct a unified `ForceConstants` model through **finite differences** or **force fitting**. The model can then be used for constraint projections, finite-temperature renormalization, and lattice-dynamics calculations.

!!! tip "Two main workflows"

    **Finite differences**: structure → displaced configurations → forces → `ForceConstants`

    **Force fitting**: structures + sampled configurations + forces → `ForceConstants`

## Get started

For a first use of MLFCS, we recommend reading in this order:

1. [Core concepts](core-concepts.md)
   Learn about `ClusterSpace`, `Supercell`, `ClusterMap`, and structural identifiability.
2. Choose how to construct force constants:
   - [Finite differences](finite-difference.md): derive force constants from displaced configurations and atomic forces.
   - [Force fitting](fitting.md): fit force constants from a set of structures and forces.
3. [Force constants](force-constants.md)
   Inspect, constrain, transform, and export the `ForceConstants` model.

## Advanced topics

### Finite temperature

- [SCPH](scph.md): self-consistent phonon calculations and finite-temperature renormalization.
- [SSCHA](sscha.md): stochastic finite-temperature effective harmonic models.

### Polar materials

- [Long-range forces](long-range-forces.md): separate and combine dipole long-range interactions and short-range force constants.

### Common questions

- [Q&A](Q&A.md): force storage, atom ordering, structural identifiability, solvers, and common parameter choices.

## Core objects

The basic relationships between MLFCS objects are:

```text
Primitive-cell ASE Atoms ──→ ClusterSpace ──┐
                                            ├──→ ClusterMap → finite differences/fitting → ForceConstants
Supercell ASE Atoms ───────→ Supercell ─────┘
```

In this flow:

- `Atoms` describes the crystal structure;
- `ClusterSpace` defines the force-constant space to model;
- `Supercell` defines the geometry used for displacement or force calculations;
- `ClusterMap` maps symmetry-reduced parameters onto the explicit supercell;
- `ForceConstants` stores the resulting force-constant model.

## Units and structure convention

MLFCS uses ASE `Atoms` objects as its common structure representation.

| Quantity | Unit |
| --- | --- |
| Length | Å |
| Energy | eV |
| Force | eV/Å |
| Order-$n$ force constants | eV/Å$^n$ |

!!! important "Supercell and atom order"

    The user-supplied supercell and its **atom order** define the geometry convention used for force calculations, parameter mapping, and force-constant reconstruction. Preserve this order throughout the workflow.
