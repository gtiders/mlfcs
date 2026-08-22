---
title: Relationship to other software
audience:
  - beginner
status: stable
code_verified: 4.0.0a4
---

# Relationship to other software

MLFCS sits between atomic-force generation and phonon, anharmonic, or transport post-processing. It does not replace electronic-structure programs or every downstream solver; it converts structures and forces into symmetry-reduced, constrained, and validated force constants.

```text
DFT / MLIP / ASE Calculator
            │ structures + forces
            ▼
          MLFCS
 finite difference / fitting
 symmetry / constraints / IFC
            │ validated force constants
            ▼
phonopy / phono3py / ShengBTE / ALAMODE
```

## ASE: the public structure and calculator boundary

MLFCS uses ASE `Atoms` for primitives, reference supercells, displaced structures, and snapshots, and receives direct forces through ASE `Calculator`. ASE supplies general structure containers, file I/O, and the calculator protocol; MLFCS supplies periodic relations, interactions, orbits, displacement plans, fitting, and IFC construction.

MLFCS is therefore not tied to one DFT code or ML potential. Direct calculators can drive finite differences and SSCHA; external VASP or Quantum ESPRESSO jobs can use `sow()` and `reap()`.

## phonopy: harmonic phonons and structure conventions

phonopy provides mature harmonic dynamical matrices, band structures, densities of states, and supercell workflows. MLFCS can produce phonopy-compatible FC2, but harmonic analysis remains phonopy's responsibility.

The most reliable workflow establishes the phonopy primitive and supercell first. Phonopy FC2 is dense on a target supercell, whereas native MLFCS IFCs use primitive sites and exact integer translations, so export performs an explicit target realization.

## phono3py: three-phonon transport

phono3py evaluates FC3-driven scattering, lifetimes, and lattice thermal conductivity. MLFCS constructs FC2/FC3 and writes compatible HDF5; q meshes, scattering processes, and BTE solving remain in phono3py.

External HDF5 stores dense target-supercell arrays rather than the native exact-$R$ interaction model. Compare physical arrays on the same target, not incidental block ordering.

## ShengBTE: text IFCs and the BTE

ShengBTE consumes format-specific FC2, FC3, and optional FC4 text. The MLFCS writer converts primitive labels, integer translations, Cartesian components, and units, but does not invent the physical settings in `CONTROL`.

Block order is not a unique physical identity. Valid files must be compared after aligning atom labels, translations, and tensor components.

## ALAMODE: FCSXML and post-processing

ALAMODE's `alm` fits force constants and `anphon` performs phonon and anharmonic analysis. Some capabilities overlap, but the internal representations and fitting bases differ.

MLFCS writes FC2–FC4 FCSXML. ALAMODE's 27-image encoding is handled only at the writer boundary and does not define MLFCS interaction, cutoff, or fitting semantics.

## hiPhive: a related problem with a different model

hiPhive also uses clusters, orbits, symmetry reduction, and force fitting, and provides a ForceConstantPotential. MLFCS shares general mathematics such as representative interactions and orbit expansion but does not copy its cluster-space implementation.

MLFCS uses primitive exact-$R$ IFCs, one-reference fitting, Wick fitting coordinates, and Taylor physical output. A hiPhive FCP is designed to evaluate forces as a potential; MLFCS `ForceConstants` is primarily physical IFC data.

## DFT and machine-learning potentials: force sources

Electronic-structure codes and ML potentials provide forces for requested structures. MLFCS does not control their accuracy settings and cannot use fitting to repair inconsistent parameters, atom order, or unconverged forces.

A reproducible calculation retains structures, calculator settings, raw forces and units, configuration IDs, atom order, MLFCS version, cutoffs, body orders, and constraints.

## Where to start

If the final consumer is phonopy, phono3py, ShengBTE, or ALAMODE, establish its primitive, supercell, and units before creating the MLFCS reference. Starting from the consumer's structures reduces ambiguity at export.

Continue with [structures and reference frames](../concepts/structures.md), the [interoperability overview](../interoperability/index.md), and [structure conventions](../interoperability/structure-conventions.md).
