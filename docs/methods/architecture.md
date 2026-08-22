# MLFCS 4.0 technical overview

English | [中文]

## 1. Scope and design goals

The base MLFCS pipeline is a clean ASE-first implementation for reconstructing force constants
from user-supplied forces. It does not embed a force calculator and has no command-line interface.
The native `mlfcs.physics.sscha` module combines compact-FC2
q-space sampling with the same Gram fitting parameterization for finite-temperature effective
FC2 calculations. The main public workflow remains a Python API:

```python
calculation = ForceConstantCalculation(
    atoms,
    reference=reference_supercell,
    order=4,
    cutoff=-3,
)

structures = calculation.sow()
forces = obtain_forces(structures)
force_constants = calculation.reap(forces)
```

An ASE `Calculator` can optionally be passed to `run`, but calculator selection and ownership
remain the user's responsibility. The principal goals of the rewrite are:

- one parameterized algorithm for second and higher orders;
- deterministic structure and configuration ordering;
- sparse reconstruction without allocating full high-order tensors;
- strict translational acoustic sum-rule constraints;
- explicit, extensible output formats;
- serial, checkpoint-friendly calculator evaluation;
- controlled CPU or GPU selection for JAX;
- black-box verification against the previous implementation and external readers.

## 2. Package architecture

The implementation is separated by responsibility:

```text
src/mlfcs/
  structure/                     lattice arithmetic, geometry, mapping, symmetry, q grids
  interactions/                  exact-R keys, tensor actions, orbits, enumeration, realization
  basis/                         Wick basis and Wick-to-Taylor intertwiners
  force_constants/               canonical IFC data, expansion, dense and target views
  constraints/                   translational and rotational physical constraints
  finite_difference/             stencils, sampling, reconstruction and calculation workflow
  fitting/                       datasets, design kernels, Gram systems and fitting workflow
  physics/                       harmonic utilities, SCPH and SSCHA workflows
  io/                            native and external force-constant formats
```

The dependency graph is deliberately not the same as the runtime sequence:

```text
structure -> interactions -> basis
structure + interactions -> force_constants
lower representations -> constraints / finite_difference / fitting
force_constants + fitting -> physics
structure + force_constants -> io
```

`io` and `physics` are terminal layers. Core data objects never import writers or workflows.

The former compatibility-only modules have been removed. New code imports from `mlfcs` or
the responsibility-specific packages. This replaces duplicated order-specific implementation
with a shared pipeline; order-dependent behavior is still expressed through `order`, tensor rank,
permutations, and recursive finite-difference keys.

The refactor is intentionally numerical-neutral. It does not change algorithms, defaults,
constraint equations, sparse labels, HDF5 schema, or output semantics. The regression suite
checks representative IFC tensors, residuals, frequencies, and import boundaries before and
after module moves.

## Public signatures

The supported entry points are grouped in `mlfcs.finite_difference.calculation`,
`mlfcs.fitting`, `mlfcs.constraints`, `mlfcs.physics.scph`, and
`mlfcs.io`. Their callable signatures are declared at the definition site, including
keyword-only options and defaults. Fitting and SSCHA are loaded lazily so importing the base
finite-difference or IO API does not initialize JAX.

The SSCHA package is an explicit submodule but has no additional runtime dependency. Applications
opt in with `from mlfcs.physics.sscha import SSCHA`.

## 3. Generic force-constant pipeline

### 3.1 Deterministic supercells

ASE `Atoms` is the structure model. Each calculation uses an explicit `StructureRelation` between
the primitive and a user-ordered reference supercell; general integer `3x3` supercell matrices
are supported. The reference order is never replaced by a hidden cell-major representation. Each
supercell stores:

- `primitive_index`;
- `cell_translation`;
- `primitive_scaled_position`.

The explicit maps avoid reconstructing atom identity from floating-point positions. `PeriodicIndex`
looks up atoms by `(primitive_site, translation_residue)` in the general lattice quotient.

### 3.2 Neighbor-shell cutoffs

A negative integer cutoff selects a neighbor shell. Distances are computed with ASE minimum-image
conventions. Nearly split but physically equivalent shells are merged with tolerances compatible
with the previous implementation (`rtol=1e-5`, `atol=1e-8`). This is important for relaxed
multi-species structures whose equivalent distances can differ at approximately `1e-6` angstrom.

For negative cutoffs, MLFCS reports two distinct quantities. The supercell-limit line gives the
maximum shell and cutoff radius enumerable by the current supercell. The selected-cutoff line
gives the user-requested shell and the radius actually passed to cluster enumeration. A request
beyond the shell limit is rejected instead of silently reusing the farthest available distance.

### 3.3 Symmetry-reduced cluster orbits

The orbit engine is parameterized by order. It combines:

- cluster enumeration within the cutoff;
- space-group atom permutations from spglib;
- all relevant force-constant axis permutations;
- Cartesian tensor rotations;
- stabilizer constraints;
- independent tensor bases and pivot components.

Cluster discovery works on primitive sites plus exact integer translations. It recursively
generates nondecreasing neighbor multisets and rejects a prefix as soon as one pair violates the
all-pair cutoff. Space-group operations act affinely on these exact labels, after which every image
is re-anchored so its first translation is zero. Equal-site Cartesian axes are
reduced in a label-symmetric basis before stabilizer constraints are accumulated. This preserves
the FC2--FC4 orbit subspaces while preventing repeated-site FC5/FC6 stabilizers from allocating
the full square `3**order` action matrix.

A tensor action is kept as a rotation plus an axis permutation. Stabilizers are contracted with
the compressed basis by matrix-free NumPy tensor operations instead of constructing every dense
`3**order x 3**order` transformation matrix.

### 3.4 Recursive finite differences

An order-`n` force constant is obtained from an `(n-1)`-fold force derivative. Central-difference
sign combinations are generated recursively, so the same stencil machinery covers every
supported order. The nominal sign count grows as `2**(order-1)`. Interaction symmetry reduces
the displacement keys, but every surviving key currently evaluates its complete signed stencil.
A further signed-configuration orbit reduction has been derived but is not part of the
production path.

Every sow structure contains a stable zero-based configuration ID, atom-order label,
and displacement array. Positional reap follows the exact sow order. Mapping-based reap accepts
results in arbitrary arrival order when keyed by configuration ID.

### 3.5 Direct-calculator zero-step extrapolation

The optional extrapolation backend builds several complete central-difference subplans around the
configured displacement, contracts each independently, and fits the derivatives as a polynomial
in `h^2`. Its zero-step intercept enters the same reconstruction and sum-rule pipeline. This
backend is restricted to direct serial ASE Calculator execution; external `sow()` / `reap()` keeps
one deterministic displacement plan.

### 3.6 Sparse reconstruction

The reconstructed result is stored as:

```text
sites:         (number_of_images, order)
translations:  (number_of_images, order - 1, 3)
tensors:       (number_of_images, 3, ..., 3)
```

It is not stored as the full

```text
(n_primitive, n_supercell, ..., n_supercell, 3, ..., 3)
```

array. These labels do not contain source-reference atom indices. Dense materialization into a
chosen target supercell is lazy. When an estimated dense allocation exceeds the configured
budget, MLFCS emits a `RuntimeWarning` but leaves the final decision to the user. Sparse HDF5
writing never requires this dense allocation.

## 4. Numerical and programming techniques

### 4.1 JAX fitting kernels

JAX is intentionally limited to the dense Wick force-design kernel used by the joint fitter. It
uses batched contractions and JIT-compatible array operations; orbit enumeration, tensor actions,
finite-difference stencils, sparse constraints, reconstruction, and I/O remain NumPy/SciPy host
work. Double precision is enabled for fitting accuracy.

The fitter's `jax_platform` option accepts `"auto"`, `"cpu"`, or `"gpu"` and selects an explicit
device without changing JAX's process-global backend selection. Static interaction buffers and
compiled kernels are prepared once per fit. GPU reduction uses bounded device-side sparse chunks,
so the physical design matrix is not copied through the host before Gram accumulation.

### 4.2 Matrix-free symmetry transformations

The old-style approach of materializing large tensor representation matrices becomes expensive
as `3**order` grows. MLFCS stores the underlying Cartesian rotation and permutation, then applies
them through tensor contractions. A dense action matrix remains available only when a small
linear-algebra operation explicitly requires it.

### 4.3 Sparse linear algebra

Reconstruction and ASR use SciPy sparse matrices. Large rectangular constraint systems are not
passed to a full SVD. This avoids the failure mode where `full_matrices=True` creates an enormous
left-singular-vector matrix unrelated to the small number of unknown parameters.

For sum-rule projection, MLFCS uses sparse LSMR for every parameter-space size. It computes the
minimum-norm correction without forming `A.T @ A`, avoiding quadratic Gram storage and squared
conditioning.

Finite-difference reconstruction and force fitting share the same interaction space, translational
constraint builder, residual metric, and final LSMR projection. Their top-level solvers remain
different mathematical problems: finite differences project an already reconstructed parameter
vector, whereas fitting minimizes a force residual inside the constraint null space. The current
fitting null-space projector forms dense `C @ C.T`; replacing it automatically for large constraint
systems is tracked in the roadmap.

The final convergence test is relative to the parameter and constraint scale, not a fixed
absolute residual alone.

### 4.4 Strict acoustic sum rule

For an order-`n` force constant, translational invariance requires

```text
sum over one atom axis of Phi(i1, ..., in) = 0
```

with all other atom and Cartesian indices fixed. Permutation symmetry makes equivalent atom axes
redundant. MLFCS constructs this constraint in the independent orbit-parameter space and projects
onto its null space.

This agrees with the physical constraint used by ALAMODE and hiphive. It differs from the
previous fourth-order implementation, which summed two atom axes together, and from the previous
relative-weight compensation that did not strictly impose `A p = 0` and could amplify large
components.

### 4.5 Memory-aware data flow

The major memory techniques are:

- reconstruct only symmetry-generated cluster images;
- retain sparse tensors through HDF5 output;
- avoid dense high-order action matrices where possible;
- use matrix-free sparse LSMR rather than Gram matrices or full rectangular SVDs;
- evaluate ASE calculator configurations serially;
- expose `evaluate()` so forces can be checkpointed before reconstruction;
- warn before dense materialization rather than unexpectedly allocating silently.

The order-5 NaS smoke calculation demonstrates why this matters. Its sparse representation used
roughly 1.06 GiB peak memory, while the equivalent dense array would require approximately
243 GiB.

## 5. I/O and ordering

I/O is selected explicitly through `format`:

- `hdf5`: generic dense or sparse storage for any order;
- `shengbte`: scientific-notation text for orders 3 and 4;
- `phonopy`: full dense second-order text format;
- `phonopy_hdf5`: streamed full-supercell FC2 HDF5;
- `phono3py_hdf5`: streamed full-supercell FC3 HDF5.
- `alamode`: combined FC2--FC4 ALAMODE FCSXML with explicit MLFCS atom mappings.

The phonopy writer expands compact FC2 to `(N, N, 3, 3)`, applies translational equivalence to
every first supercell atom, and preserves the explicit reference-supercell order on both atom
axes. A generated K3Au3Sb2 3x3x3 file was successfully
read by phonopy's own parser as `(216, 216, 3, 3)` with a maximum ASR residual of
`2.60e-14`.

The phonopy and phono3py HDF5 writers preserve reference order at the format boundary and stream
one first-atom slab at a time. This avoids constructing a second complete
full-supercell tensor solely for output.

ShengBTE output is cluster-and-translation based rather than a simple global supercell tensor
order. The writer derives its primitive atom indices and lattice translations from the verified
reference-frame lattice labels and uses scientific notation for both supported orders.

The ALAMODE adapter preserves the explicit reference-supercell order and serializes the existing
`primitive_index`/`cell_translation` mapping rather than rediscovering a primitive cell. Its
mirror-cell identifiers use the fixed 27-image convention required by the ALAMODE output layer. Export first tries an
equivalent Minkowski-reduced supercell basis, then rejects a geometry whose true minimum image
remains outside that representable set.

## 6. Comparison with the previous implementation

| Area | Previous implementation | MLFCS 4.0 |
|---|---|---|
| Public interface | Order-specific CLI workflows | Pure Python ASE API |
| Code organization | Separate third/fourth-order implementations | Shared order-parameterized pipeline |
| Calculator | Workflow-integrated calculator handling | User-owned ASE Calculator or external forces |
| Supercell identity | Position/order conventions spread across workflow | Explicit primitive and translation arrays |
| Sow/reap contract | Primarily file and positional convention | IDs, positional or mapping reap |
| Higher orders | Separate fixed-order logic | Generic reconstruction validated through order 5 |
| Tensor actions | Dense transformations in critical paths | Matrix-free rotation and permutation kernels |
| Reconstruction storage | Primarily dense order-specific arrays | Sparse cluster images with lazy dense conversion |
| ASR FC3 | One-axis sum with relative compensation | Strict one-axis constrained projection |
| ASR FC4 | Incorrect two-axis sum | Same physical one-axis rule as every order |
| Large constraints | Full SVD could allocate huge matrices | Sparse LSMR projection; scalable fitting null space is on the roadmap |
| JAX | Not a primary backend | Explicit CPU/GPU tensor backend |
| Output | Order-specific writers and CLI conventions | Explicit extensible `format` routing |
| Phonopy FC2 | Dependency-driven interface | Dependency-free compatible dense writer |
| Dense memory policy | Allocation determined by order-specific path | Estimate, warning, and sparse alternative |

The rewrite intentionally does not reproduce the previous projected FC3/FC4 values when the old
ASR operation changes the physical constraint. Compatibility is instead tested at the geometry,
neighbor-shell, orbit-plan, raw reconstruction, atom-order, and file-format levels.

## 7. Validation and measured behavior

All performance measurements below were run serially to avoid concurrent memory pressure.

- Si 2x2x2, third order: 11 orbits, 72 configurations, strict ASR maximum residual
  `3.89e-15`.
- Si 2x2x2, fourth order: 41 orbits, 1056 configurations, strict ASR maximum residual
  `2.54e-14`.
- Strict FC4 versus previous FC4 on the saved Si force set: maximum difference `1.31e-3`, RMS
  `5.71e-5`; this is an ASR-method difference rather than ordering error.
- K3Au3Sb2, KAsPt, and NaS fourth-order `-3` plans match the previous cutoff, irreducible-cluster
  count, and configuration count exactly.
- NaS fourth-order end-to-end peak memory decreased from approximately 4.92 GiB to 1.14 GiB.
  The measured wall time changed from approximately 6 min 57 s to 7 min 44 s, reflecting a
  deliberate memory-versus-runtime tradeoff in the current sparse Python/JAX path.
- An earlier Si fourth-order old path reached approximately 9.51 GiB; the strict sparse path is
  approximately 1.08 GiB.
- NaS order 5, 2x2x2, first shell: 16 orbits, 403 independent parameters, 2432 force
  configurations, and 1686 sparse cluster images. The HDF5 result is about 789 KiB.
- K3Au3Sb2 FC2, 3x3x3, sixth shell: 216 atoms, 30 orbits, 24 force configurations, and about
  470 MiB peak memory with the NEP calculator.

The complex-material neighbor-shell regression results at fourth order and cutoff `-3` are:

| Material | Irreducible clusters | Force configurations |
|---|---:|---:|
| K3Au3Sb2 | 61 | 3568 |
| KAsPt | 45 | 2936 |
| NaS | 43 | 2016 |

## 8. Native stochastic effective-harmonic module

MLFCS implements the effective-harmonic loop entirely with native components:

1. generate small random Cartesian displacements when no initial FC2 is available;
2. evaluate arbitrary ASE forces and optionally energies;
3. fit symmetry-reduced compact FC2 with the streamed-Gram fitter;
4. sample the quantum or classical canonical ensemble at commensurate q points;
5. repeat the force evaluation and FC2 fit for the requested number of updates.

Both direct serial Calculator execution and iteration-level `sow/reap` are supported. The latter
keeps external scheduling and concurrency under user control. Each fit produces an immutable
result containing FC2, sampling mode, energy averages, free energy, and its finite-sampling
standard error and q-space sampling diagnostics. The active FC2 may be replaced by an average of
the final iterations and written through MLFCS's phonopy-compatible text or HDF5 writers.

This module estimates a temperature-dependent effective harmonic Hamiltonian from thermal force
samples. It is not the explicit FC3 bubble or FC4 loop implementation discussed for ALAMODE.
Detailed usage, formulas, ordering rules, and stability guidance are in
[`SSCHA.md`].

The SSCHA tests use analytic harmonic models for classical variance, quantum zero-point motion,
translation removal, clipping diagnostics, Cartesian initialization, canonical sampling,
external ID-mapped reap, repeated native Gram fits, FC2 averaging, free energy, and HDF5 output.
An independent development-only phonopy reference checks q-space frequencies and statistics.

## 9. Current limitations and likely next steps

- The generic combinatorial machinery supports higher orders, but cost still grows through
  cluster combinations, `order!` permutations, `3**order` tensor components, and
  `2**(order-1)` finite-difference signs.
- ShengBTE export is intentionally restricted to orders 3 and 4; generic higher-order output
  should use sparse HDF5.
- Non-analytic long-range electrostatic corrections are not implemented.
- Explicit diagrammatic FC3 bubble and FC4 loop self-energies are not implemented. The optional
  SSCHA module instead obtains temperature-renormalized FC2 from stochastic thermal force data.
- Molecular-dynamics effective harmonic fitting remains a separate possible extension.
- SSCHA convergence is deliberately controlled by the caller through repeated `step()` calls;
  the library does not yet define a universal automatic stopping criterion.

## 10. Version summary

`v3.0.0` consolidates phonopy FC2 export, explicit neighbor-shell diagnostics,
generic sparse reconstruction, strict ASR, and the independent optional `mlfcs.physics.sscha` module.
It also includes a redesigned ASE-first direct and
external API, structured iteration history, free-energy uncertainty, final-iteration averaging,
and phonopy-native FC2 output.
