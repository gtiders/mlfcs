---
title: "MLFCS: symmetry-reduced finite-difference force constants with ASE and JAX"
tags:
  - Python
  - force constants
  - anharmonic phonons
  - finite differences
  - materials science
authors:
  - name: gtiders
    affiliation: "1"
affiliations:
  - name: Independent Researcher
    index: 1
date: 3 August 2026
bibliography: paper.bib
---

# Summary

Interatomic force constants describe how the energy of a crystal changes when its atoms move.
They are the central input to harmonic phonon calculations and to anharmonic calculations of
thermal expansion, phonon scattering, and lattice thermal conductivity. MLFCS is a Python library
that reconstructs these tensors from atomic forces. A user supplies an ASE structure and either an
ASE calculator or forces evaluated by an external workflow; MLFCS handles symmetry reduction,
finite-displacement generation, reconstruction, optional translational constraints, and export.

Unlike order-specific displacement programs, MLFCS uses one parameterized implementation for
second and arbitrarily higher orders. Third- and fourth-order calculations are the primary
production paths, while higher-order tensors remain available through a sparse HDF5
representation. The package supports CPU execution and JAX-accelerated Cartesian tensor
operations on compatible GPUs [@jax2018].

# Statement of need

Finite-displacement force constants require more than repeatedly moving atoms. Equivalent atomic
clusters and Cartesian tensor components must be identified under crystal symmetry and index
permutations; displacement configurations must be reproducible; periodic images and atom order
must remain consistent from force generation through export; and the rapidly increasing tensor
size must not force premature dense allocation. These concerns become particularly visible for
fourth and higher orders and for multicomponent cells.

Established packages address important parts of this problem. phonopy and phono3py provide widely
used harmonic and third-order workflows [@phonopy2015; @phono3py2015]. ALAMODE provides
displacement- and regression-based anharmonic lattice-dynamics methods [@alamode2018], while
hiphive builds force-constant models through regression in symmetry-adapted cluster spaces
[@hiphive2019]. ShengBTE consumes second- and third-order force constants for phonon Boltzmann
transport [@shengbte2014], and FourPhonon extends that workflow to fourth-order interactions and
four-phonon scattering [@fourphonon2022]. MLFCS serves users who specifically need deterministic,
central finite differences
from arbitrary ASE-compatible force providers, a direct order parameter extending beyond FC3,
and a sparse result that can be exported without coupling force generation to a particular
electronic-structure program or potential.

# State of the field

MLFCS is complementary to, rather than a replacement for, phonopy, phono3py, ALAMODE, or hiphive.
It adopts ASE [@ase2017] as the calculator boundary, allowing classical, machine-learning, and
first-principles adapters to share the same calculation object. Its `sow()` and `reap()` contract
also supports schedulers and electronic-structure calculations that cannot run inside Python.
Stable configuration identifiers and a plan hash detect missing, duplicated, reordered, or stale
force datasets.

The distinctive contribution is the combination of a recursive order-independent central-
difference plan, symmetry-reduced cluster tensors, and sparse reconstruction. ShengBTE export is
provided for FC3 and FC4, including inputs used by ShengBTE and FourPhonon; dense phonopy output
is provided for FC2, phonopy/phono3py HDF5 interoperability for
FC2/FC3, and a generic HDF5 schema for higher orders. A compatibility mode reproduces the
historical thirdorder periodic-image convention, while the default export uses the same
symmetry-closed support as reconstruction.

# Software design

For order $n$, MLFCS treats a force constant as an $(n-1)$-fold displacement derivative of force.
A recursive centered stencil generates signed displacement keys. Space-group operations,
permutations of force-constant indices, and cluster stabilizers reduce each cluster tensor to its
independent Cartesian parameters. Only required force derivatives are sampled, after which the
parameters are expanded to symmetry-related sparse cluster images.

The acoustic sum rule is imposed as a constrained projection in the independent orbit-parameter
space rather than by relative post-hoc tensor corrections. Small systems use a Gram-matrix null
space with sparse LSMR refinement; large systems use a matrix-free sparse LSMR projection. Dense
tensors are materialized only on request. Contiguous sparse arrays, batched JAX transformations,
JIT compilation, displacement-key deduplication, and streamed external output limit time and
peak memory.

The API deliberately does not own the force calculator. `run()` evaluates a user-owned ASE
calculator serially to avoid multiplying the memory of large machine-learning models. `sow()`
returns an ordered list of ASE structures for external evaluation, and `reap()` validates and
reconstructs returned forces. This separation keeps scientific provenance and parallel execution
under user control.

# Research impact statement

MLFCS has been publicly developed and used for force-constant and thermal-transport workflows.
Version 3.0 adds a reproducible test hierarchy rather than relying on agreement with the previous
implementation. FC2 and FC3 are compared with independent phonopy and phono3py results for both
binary AlN and multicomponent K4As4Pt2 models. A 3x3x3 silicon dataset checks the complete external
sow/reap and ShengBTE path against VASP forces. FC4 is tested against a separate JAX fourth
derivative of an analytic FCC Morse energy; halving the displacement produces the expected
second-order reduction in finite-difference error. Provenance files, checksums, atom-order
adapters, and explicit ASR-on/off comparisons accompany the reference data.

Continuous integration runs the public API on Python 3.12 and 3.13, builds distributions, and
executes scientific references serially to control memory. These tests establish a foundation for
reusing force constants in transport, anharmonic phonon, and temperature-dependent effective
harmonic studies while keeping the generating potential replaceable.

# AI usage disclosure

Generative AI tools were used during the version 3 refactoring to assist code drafting,
documentation, test organization, and preparation of this paper. The maintainer reviewed all
changes. Numerical claims were checked with executable tests, independent reference data,
analytic differentiation where available, and continuous integration; generated text was checked
against the implementation and cited sources.

# Acknowledgements

The author thanks the developers of ASE, JAX, phonopy, phono3py, spglib, hiphive, ALAMODE, NumPy,
and SciPy. No external funding is declared in this draft.

# References
