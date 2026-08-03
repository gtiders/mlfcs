---
title: "MLFCS: arbitrary-order finite-difference force constants with ASE and JAX"
tags:
  - Python
  - force constants
  - anharmonic phonons
  - finite differences
  - materials science
authors:
  - given-names: Yibo
    surname: Gao
    corresponding: true
    affiliation: "1"
affiliations:
  - name: Central South University, China
    index: 1
date: 3 August 2026
bibliography: paper.bib
---

# Summary

Interatomic force constants quantify how the energy of a crystal changes when its atoms move.
Second-order force constants determine harmonic phonons, while third- and fourth-order terms are
needed for quantities such as phonon scattering rates and lattice thermal conductivity. Obtaining
these tensors by finite differences is conceptually simple—displace atoms and calculate
forces—but becomes computationally demanding because the number of atomic clusters, Cartesian
components, and displacement combinations grows rapidly with order.

MLFCS is an open-source Python library for reconstructing symmetry-reduced force constants from
second to arbitrary order. It accepts structures through the Atomic Simulation Environment (ASE)
and receives forces either from a user-selected ASE calculator or an external electronic-
structure workflow [@ase2017]. A single order-parameterized pipeline performs periodic cluster
enumeration, symmetry reduction, recursive central finite differences, sparse reconstruction,
optional enforcement of translational invariance, and output to established phonon formats.
Third- and fourth-order calculations are the principal production paths; higher-order results use
the same algorithm and can be retained in a generic sparse HDF5 representation.

# Statement of need

High-order finite-difference calculations present three connected problems. First, space-group
symmetry, permutations of force-constant indices, and tensor stabilizers must be applied
consistently so that only independent quantities are sampled. Second, atom identities and
periodic images must remain unambiguous from displacement generation to force collection and
file export. Third, a nominally straightforward implementation can become limited by memory
before the force calculations begin: a dense order-$n$ Cartesian action scales as
$3^n \times 3^n$, a dense force-constant array grows with multiple powers of the supercell atom
count, and a full singular-value decomposition can allocate matrices unrelated to the number of
independent parameters.

These limitations are especially important when the force provider is a machine-learning
potential. Fast force evaluation exposes symmetry processing and tensor reconstruction as a
significant fraction of the total runtime, while loading several calculator instances in
parallel can unnecessarily multiply memory use. MLFCS therefore treats resource efficiency as a
scientific-enabling requirement rather than an implementation detail. It stores force constants
as sparse symmetry-generated cluster tensors, applies Cartesian rotations without materializing
high-rank representation matrices, deduplicates equivalent displacement tasks, streams large
format conversions, and evaluates an in-process ASE calculator serially by default. JAX provides
JIT-compiled and batched tensor contractions on CPUs or compatible GPUs [@jax2018], while large
sparse constraints remain on memory-efficient CPU solvers.

# State of the field

phonopy and phono3py provide widely used harmonic and third-order finite-displacement workflows
[@phonopy2015; @phono3py2015]. ALAMODE supports displacement and regression approaches for
anharmonic lattice dynamics [@alamode2018], and hiphive fits force-constant models in
symmetry-adapted cluster spaces [@hiphive2019]. ShengBTE consumes second- and third-order force
constants for phonon Boltzmann transport [@shengbte2014], while FourPhonon adds fourth-order
interactions and four-phonon scattering [@fourphonon2022].

MLFCS is complementary to these packages. Its specific role is deterministic central finite
differences from an arbitrary ASE-compatible force source, expressed through one order parameter
rather than separate implementations for each tensor rank. The calculation remains independent
of a particular electronic-structure code or machine-learning potential, and sparse results can
be retained without forcing conversion to a dense array. MLFCS exports dense phonopy FC2,
phonopy/phono3py HDF5 for FC2/FC3, and ShengBTE-compatible FC3/FC4, while generic HDF5 preserves
higher orders. The library also provides an optional stochastic effective-harmonic module for
temperature-dependent FC2; this module is separate from the finite-difference reconstruction.

# Software design

For force-constant order $n$, MLFCS evaluates an $(n-1)$-fold displacement derivative of force.
A recursive centered stencil produces signed displacement keys. Space-group operations are
combined with all relevant index permutations and Cartesian tensor rotations to construct
orbits of equivalent atomic clusters. Stabilizer constraints determine a basis of independent
Cartesian components for each orbit. After the requested forces are returned, the sampled
derivatives are reconstructed directly into sparse cluster images.

The acoustic sum rule is imposed as a constrained projection in this independent parameter
space. If $p$ contains the orbit parameters and $A$ is the translational-constraint matrix,
MLFCS finds the nearest admissible solution satisfying

$$A p = 0.$$

For moderate parameter counts, a small Gram matrix identifies the null space and sparse LSMR
refines the result against the original constraints. Larger systems use sparse LSMR directly,
avoiding a dense decomposition. Dense materialization is explicit and preceded by an allocation
estimate; sparse HDF5 output never requires the complete high-order array.

Reproducibility is part of the data model. Each displaced structure carries a stable
configuration identifier, atom-order label, displacement array, and hash of the complete
displacement plan. Returned forces can follow the exact generated order or be supplied as an
identifier-keyed mapping. Missing, duplicated, reordered, or stale force sets are detected before
reconstruction. This contract permits the same calculation object to be used with a local ASE
calculator, a batch scheduler, or first-principles calculations such as VASP without embedding
site-specific execution logic in MLFCS.

# Verification and resource-aware execution

The test suite separates mathematical unit tests from independent scientific references. FC2
and FC3 are compared with phonopy and phono3py for binary AlN and multicomponent K4As4Pt2 models,
including atom-order conversion and tests with and without acoustic-sum-rule projection. A
3x3x3 silicon reference exercises the complete external VASP-force and ShengBTE-export path.
FC4 is tested without another force-constant fitter: an independent JAX implementation
differentiates an analytic FCC Morse pair energy four times, and halving the finite-difference
step produces the expected second-order error reduction.

Performance is addressed at the algorithm and operator levels rather than through whole-program
comparisons with packages that solve different fitting or finite-displacement problems. Symmetry
reduction and displacement-key deduplication avoid unnecessary force evaluations. Matrix-free
tensor actions avoid constructing $3^n \times 3^n$ Cartesian representation matrices, and sparse
constraint solvers avoid dense decompositions. The remaining high-rank tensor operations are
expressed as JAX transformations using JIT compilation, `vmap`, and batched contractions. The
same kernels run on CPUs and can be placed on a compatible GPU through an explicit backend
option; cluster enumeration and large sparse solves remain on the CPU.

The stored representation provides a directly reproducible scaling check without comparing
unlike end-to-end workflows. In a fifth-order NaS first-shell smoke test, the result contains
1,686 sparse cluster images and occupies about 789 KiB in HDF5, whereas the corresponding full
dense tensor has an estimated size of approximately 243 GiB. Continuous integration executes
scientific references serially and tests Python 3.12 and 3.13 independently so that correctness
checks do not rely on high-memory parallel execution.

# Research impact statement

MLFCS makes high-order finite differences practical for researchers who already have a force
provider but need a reproducible route from displaced structures to interoperable force
constants. The same public workflow supports classical potentials, machine-learning potentials,
and externally scheduled first-principles calculations. Its order-independent representation is
also a foundation for studying higher-order interactions without creating another fixed-order
code path. Provenance records, frozen force datasets, checksums, and independently readable
exports make numerical results auditable and reusable in downstream phonon-transport workflows.

# AI usage disclosure

Generative AI tools assisted code drafting, refactoring, documentation, test organization, and
manuscript preparation. The author reviewed all changes. Scientific claims were checked with
executable tests, independently generated reference data, analytic differentiation where
available, and continuous integration. Generated prose was checked against the implementation
and cited literature.

# Acknowledgements

The author acknowledges Central South University for institutional support and thanks the
author's research group for providing computational resources and a test dataset used during
software verification. The author also thanks the developers and maintainers of ASE, JAX,
NumPy, SciPy, spglib, phonopy, phono3py, hiphive, ALAMODE, ShengBTE, and FourPhonon. This work
received no external financial support, and no sponsor had a role in the software design,
validation, manuscript preparation, or decision to submit. The author declares no competing
interests.

# References
