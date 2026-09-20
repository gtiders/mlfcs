---
title: Orbit-Search Thresholds and Exactness
audience:
  - advanced
  - developer
status: research
code_verified: 4.0.0a6
---

# Orbit-Search Thresholds and Exactness

This note audits everything that runs *before* a fit or a finite-difference run: the lattice
relation, the primitive orbit search, and the realization of primitive orbits in a finite
reference. It lists the numerical thresholds in that path and the decisions that are currently
made with floating point or fixed-width integers although they are exact algebraic statements.

## Stage map

```
Atoms
 ├─ StructureRelation.from_atoms                 structure/relation.py            tolerance 1e-5
 ├─ normalize_supercell_matrix                   structure/integer_lattice.py     atol 1e-10, int64 guards
 ├─ PeriodicIndex                                structure/supercell_mapping.py   int64 residue keys
 ├─ PrimitiveSymmetryOperations.from_atoms        structure/symmetry.py           spglib symprec
 └─ SymmetryOperations.from_primitive_operations  integer modular arithmetic
        ↓
 InteractionSpace.from_frame                     interactions/space.py           symprec forwarded
        ↓
 build_primitive_interaction_space               interactions/primitive/builder.py      lattice-frame integers
 ├─ traverse_indexed_orbit                        interactions/algebra/indexed_orbit.py  exact stabilizer actions
 ├─ invariant_kernel                              interactions/algebra/invariants.py     modular rank + verified kernel
 ├─ select_independent_rows                       interactions/algebra/invariants.py     QR search, certified count
 └─ normalize_pivot_basis                         interactions/algebra/invariants.py     float solve of the Cartesian image
        ↓
 realize_interaction_space +
 validate_realization_identifiability             interactions/realization.py      exact lattice ranks
        ↓
 ReciprocalQuotientGrid                           structure/reciprocal.py          atol 1e-12
```

## Thresholds in this stage

| Site | Value | What it decides | Kind |
| --- | --- | --- | --- |
| `space.py`, `symmetry.py` | `symprec=1e-5` $(\text{Å})$ | spglib symmetry precision | genuine, keep |
| `space.py` | `symprec` again | forwarded as the *atom-mapping* tolerance | one number, two meanings |
| `relation.py` | `1e-5` | integer supercell test and atom mapping cost | float test for an integer fact |
| `relation.py` | `1e-7` | training-frame cell check | same question, 100x tighter |
| `relation.py` | `1e-5` | `align_structures` | third copy of the same value |
| `integer_lattice.py` | `atol=1e-10` | "supercell_matrix must contain integers" | float test for an integer fact |
| `reciprocal.py` | `atol=1e-12` | q-point compatibility re-check | float re-check of exact labels |
| `symmetry.py` | `symprec * 10.0` | match integer site shifts by distance | arbitrary relaxation factor |
| `candidates.py` | `+1e-8` | neighbour-list radius padding | never binds |
| `invariants.py` | `1e-9` relative | pivot-row search in the Cartesian image | search only: the pivot count is certified by the exact lattice dimension |

## Decisions that are exact algebra

| Site | Current | Exact route |
| --- | --- | --- |
| `invariant_kernel` | rank modulo a large prime, exact kernel verified over the integers | done: the constraint rows are integer spglib rotations applied to the $0/1$ label basis, the dimension is $C - \operatorname{rank}_{\mathbb{F}_p}(G)$, and the returned columns are checked with `rows @ basis == 0` |
| `select_independent_rows` | QR search, the count certified by the lattice dimension | done: no verdict depends on the threshold |
| `normalize_pivot_basis` | `np.linalg.solve` | floating point, but on the deterministic Cartesian image of an exact integer basis |
| `indexed_orbit._action_signature` | integer lattice tuple key | done |
| `indexed_orbit` stabilizer residual | integer stabilizer actions, no filter | done |
| `validate_realization_identifiability` | integer lattice basis and lattice action matrices, exact rank (modular, exact fallback) | done |
| `StructureRelation.from_atoms` | `allclose(transform, matrix)`, LAP mapping cost | rational basis change, then exact coset matching of folded fractional coordinates |
| `reciprocal.py` | `allclose(points @ matrix.T, rint(...))` | modular identity on the integer labels |
| `integer_lattice.py` | `allclose(values, rint(values), atol=1e-10)` | the caller derives the matrix exactly, so integrality is a divisibility test |

## Fixed-width integers

`int64` is not itself a problem: modular arithmetic in `int64` is exact. The cost appears where an
exact algebraic result is squeezed into a fixed-width array, which forces guards that would not
otherwise exist: the three `np.iinfo(np.int64)` overflow checks and the `_sympy_int64_matrix`
conversion in `structure/integer_lattice.py` exist only to move exact `sympy` integers into `int64`
arrays. Keeping the computation in `sympy` (or Python `int`) until the final, bounded output
(labels, translations, representatives) removes those guards and the conversions.

The label, translation, and residue keys in `interactions/keys.py`, `structure/supercell_mapping.py`
and `structure/reciprocal.py` genuinely are integers and stay integer; what changes is their
origin, which should be an exact integer computation rather than a rounded float.

## Measured effect of the first two fixes

Benchmark: the Ba8Ga16Ge30 reference (54-atom primitive, $2\times2\times2$ supercell), neighbour
shells $-8$ (FC2), $-8$ (FC3), $-5$ (FC4), bodies 2/3/4, `symprec=1e-4`. That resolves to radii
4.8746, 4.8746 and 4.0252 Å with 213, 938 and 628 orbits (1791, 21270 and 26628 parameters).

| stage variant | realization | whole stage |
| --- | --- | --- |
| tolerance filters plus `matrix_rank` (before) | 25.03 s | 28.51 s |
| exact, one `Fraction` per basis entry | 40.07 s | 43.63 s |
| exact, integer pipeline (current) | 22.64 s | 26.23 s |

Per order, realization only: 0.47 / 6.57 / 19.17 s before, 0.44 / 6.26 / 17.54 s now.

Three changes account for the difference: the basis is reconstructed exactly from its few distinct
magnitudes instead of once per entry, tensor-action matrices are cached by their exact integer
signature instead of being rebuilt per image, and both the coefficient accumulation and the
component rank stay in integer arithmetic, with one common denominator applied per component at
rank time.

## Suggested order

1. `realization.py`: exact rank and exact zero tests. The `> 1e-10` filter can silently drop a
   non-zero coefficient and flip the identifiability verdict, so this is a correctness fix first.
2. `indexed_orbit._action_signature`: integer keys instead of rounded floats. Self-contained.
3. `invariants.py`: exact kernel and rank, which requires clearing the $1/\sqrt{k}$ scaling of the
   label-symmetric basis. This removes the `tolerance` parameter from the orbit-search entry points.
4. `relation.py`: rational supercell test and exact coset matching, which removes the mapping
   tolerance from `StructureRelation.from_atoms` and leaves `symprec` to spglib alone.

`symprec` (spglib) and the neighbour cutoff radius are the only true thresholds in this stage and
should stay.

## Scaled-frame integer algebra (hiphive 1.5)

Item 3 above is written in the Cartesian frame, where a primitive cell that is not aligned with
the reference axes (fcc primitive $60^\circ$, hexagonal, rhombohedral) has action matrices with
irrational entries ($\sqrt{3}/2$, $1/\sqrt{3}$).  Orbit bases then have no exact rational form at
all, so `realization` needs a rational reconstruction and, when that reconstruction fails, a
floating-point fallback.

hiphive avoids the problem structurally by doing the symmetry algebra in the **scaled (fractional)
frame**:

| hiphive | what it does |
| --- | --- |
| `cluster_space.py:164` | `rotation_matrices` are spglib `dataset.rotations`, i.e. integer matrices in scaled coordinates |
| `core/eigentensors.py:26` | the constraint matrix is built with those integer rotations applied to the $0/1$ label-symmetric indicator tensors, so every entry is an exact integer |
| `core/utilities.py:25` | `SparseMatrix.rref_sparse` / `nullspace` solve the kernel with exact rational (sympy) elimination |
| `core/eigentensors.py:112` | `renormalize_to_integer` scales each solution by the LCM of its denominators, so the eigentensors are integers |
| `core/tensors.py:38` | `rotation_to_cart_coord(R, cell)` ($\text{cell}^T R \,\text{cell}^{-T}$) and `rotation_tensor_as_matrix` convert to Cartesian only at the boundary |
| `core/config.py:33` | the single documented escape hatch (`eigentensor_simplify_before_compress`) exists only for Cartesian rotations, e.g. hcp |

A literal port is not affordable: hiphive's `SparseMatrix` kernel over the full point group costs
70.7 ms per orbit at order 4 (cubic, 48 operations), 55.2 ms (fcc primitive) and 30.4 ms
(hexagonal), i.e. 13–44 s for our 448–628 orbit order-4 spaces, against a 0.096 s `invariants`
stage.

The frame insight does transfer, with fast integer algebra instead of sympy:

- spglib scaled rotations are exact integers for every cell tested (cubic, hexagonal, fcc primitive
  $60^\circ$), while the Cartesian ones are not;
- an integer action is exact in `int64` Kronecker contraction, and the constraint block is
  $3^{\text{order}} \times C$; the kernel dimension is $C - \operatorname{rank}_\mathbb{Q}(G)$ with
  $G = B^T B$, and a rank modulo a large prime ($2^{31}-1$) is decisive here because
  $\operatorname{rank}_{\mathbb{F}_p} \le \operatorname{rank}_\mathbb{Q}$, so a passing check is a
  proof and a failing one is detectable;
- forming the Gram (here $54 \times 54$) before ranking keeps the exact rank under 1 ms per orbit,
  whereas ranking the raw 3448-row constraint stack modulo a prime costs 100 ms;
- the cost driver is the number of distinct stabilizer actions per orbit, measured at 1.3–2.9 on
  average (maximum 19) rather than the full 48-operation point group, which puts the exact route at
  roughly 1–2 s of added work for a 448-orbit order-4 space.

Implemented for item 3, in the form the rest of the pipeline allows.  Calibrating the two frames
gives $K = (\text{cell}^T)^{\otimes\,\text{order}}$ with `symmetry.rotations[operation]` (no
transpose, composing as $\text{after} \cdot \text{before}$), which is the same map as hiphive's
`rotation_to_cart_coord`; a test pins it per generator against spglib's own Cartesian rotations.

- `PrimitiveInteractionOrbit.basis` is the exact integer basis of the invariant subspace in lattice
  coordinates, one column per fitted parameter, and the parameters are its coefficients.  No
  canonicalization happens: the pivot block of an integer subspace is rational in general, so
  demanding the identity there would force a rational parameterization in *any* frame.
- `invariant_kernel` returns that basis after dividing every column by the gcd of its entries, and
  `realization` ranks the lattice-frame realization matrix with the same basis.
- Consumers render physical tensors once, through $K$, which is a single deterministic matmul:
  `fitting/parameterization.py`, `force_constants/expansion.py`, `constraints/translational.py` and
  `finite_difference/reconstruction.py` all read `space.cell` and `orbit.basis`.

Verified numerically against the pre-change code: reconstructed finite-difference force constants
agree to a relative $2.3\times10^{-16}$ and fitted predicted forces are bit-identical; only the
parameter coordinates change, because they are coefficients in a different basis of the same
subspace.

### Why the observed components are still chosen in Cartesian

The components a finite-difference plan observes are Cartesian tensor components, and the plan must
observe $\dim$ of them whose rows determine the parameters.  That independence is an algebraic
condition on the Cartesian image, not an integer one, so it cannot be decided exactly without
symbolic arithmetic.  Measured on four cells at orders 2 and 3, choosing those rows by exact integer
greedy rank in the lattice frame instead and then inverting the Cartesian block at them:

| cell | same row set | largest condition number of the Cartesian block at the lattice rows |
| --- | --- | --- |
| cubic simple | 3/3, 2/2 | 1.0, 1.0 |
| fcc primitive | 4/4, 4/10 | 5.7, $\infty$ (singular) |
| hexagonal | 5/5, 12/12 | 4.6, 9.6 |
| SnSe | 6/6, 4/4 | 7.6, 7.6 |

An fcc primitive cell at order 3 already produces a singular block, which makes the plan
insufficient rather than merely ill-conditioned.  The selection therefore stays on the Cartesian
image, with its *count* certified by the exact lattice dimension, and it is the only numeric rank
test left in this stage.

## Threshold inventory over `src/`

The same audit over the rest of the package, grouped by what the number decides.  Values that do not
scale with the data are the fragile ones.

### Decisions that are exact algebra (float tests for integer or rational facts)

| Site | Value | Exact statement behind it |
| --- | --- | --- |
| `structure/integer_lattice.py:24` | `atol=1e-10` | the supercell matrix contains integers |
| `structure/relation.py:57` | `atol=tolerance` (the `symprec` value) | the same statement one layer up |
| `structure/relation.py:82` | `>= tolerance` | LAP atom-mapping cost is zero, i.e. an exact coset match |
| `structure/relation.py:110` | `atol=1e-7` | the training cell equals the reference cell |
| `structure/relation.py:132,148` | `tolerance=1e-5` | the aligned structure equals the reference |
| `structure/reciprocal.py:40` | `atol=1e-12` | q-point labels are integer combinations |
| `structure/symmetry.py:92` | `symprec * 10.0` | integer site-shift matching, with an invented relaxation |
| `structure/periodic_geometry.py:103,158` | `atol=1e-8 + rtol*max(...)` | tied minimum images (relative, keep) |
| `structure/periodic_geometry.py:171,176` | `rtol=1e-5, atol=1e-8` | distinct neighbour distances, which fixes shells — **flagged for reimplementation, see below** |
| `interactions/primitive/candidates.py` | removed | both heuristics lived only in the `cutoff=None` branch, which is gone; a distance or a negative shell index is required now |
| `force_constants/realization.py:64,94` | `atol=1e-7`, `> 1e-5` | the target is the same lattice and atom set |
| `io/alamode.py:64,192`, `io/shengbte.py:55` | `1e-10`, `1e-8/1e-10` | export consistency of integer or repeated content |

### Decisions that change the model (highest priority)

| Site | Value | What it decides |
| --- | --- | --- |
| `fitting/linear_solvers.py:51` | `tolerance * max(block.shape) * max(diagonal)` | the **number of free parameters** in the constraint null space |
| `fitting/constraints.py:66,74` | `1e-12`, `np.round(data, 12)` | **keep**: the rows are Cartesian components of algebraic numbers, so no integer key can decide equality; measured margin below |
| `constraints/translational.py:36` | `abs(entry) > 1e-12` | **keep**: the same guard; algebraic zeros appear at $3.5\times10^{-15}$ for a row scale of $59$, genuine coefficients at $O(10)$ |
| `fitting/gram/models.py:38` | `max(norm) * 1e-12` | which Gram columns are normalized |
| `interactions/algebra/invariants.py:30` | `1e-9` relative | observed component rows; the *count* is certified by the exact dimension |

### Thresholds that should stay

- Genuine physical thresholds: `symprec` (spglib), the neighbour cutoff radius, the finite-difference
  `displacement`, `imaginary_tolerance` and `cutoff_frequency` in THz, `mixing`, and the rotational
  sum-rule `strength`.
- Solver convergence tolerances, which should read as such: `Fitter.tolerance`, the
  `explicit_constraint_null_space` tolerance, the CG/LSMR `rtol`/`atol` in `fitting/linear_solvers.py`,
  the ASR projection tolerance in `constraints/translational.py`, `SCPH.tolerance` in THz, and the
  relative spectral cutoffs in `constraints/rotational.py` and `fitting/gram/models.py`.
- Text-export zeroing, which decides what a file looks like rather than what the model is:
  `io/numeric_text.py::zero_small_scalar` with `_TEXT_ZERO_TOLERANCE` in `io/alamode.py` and
  `io/shengbte.py`, plus `_MIRROR_TOLERANCE_BOHR`.

### Flagged: `unique_periodic_distances` decides shells through a float guard

`unique_periodic_distances(rtol=1e-5, atol=1e-8)` is the only place that turns neighbour
distances into shells, and it is called only from `candidates.py:72` when a caller asks for a
negative `cutoff` (shell index).  Its de-duplication is a float test on quantities that are
integer combinations of lattice vectors, so the shell boundaries it produces inherit a guard
that has not been justified the way the others in this note have.  Reimplementation should
either derive the shells structurally (from the seed and lattice data, not from distances) or
document the measured margin.  This is a marker, not a change: the current behaviour stays.

### Who enforces identifiability

The stage that can reject a bad choice is the realization check.  Every fit and every
finite-difference run reaches it through `InteractionSpace.realized_orbit_space`
(`interactions/space.py:164`), which calls `realize_interaction_space` and with it
`validate_realization_identifiability` (`interactions/realization.py:182`).  That function
ranks the realization matrix in the exact integer lattice frame and raises
`InteractionAliasingError` when a connected parameter component is deficient, naming the
offending clusters and both remedies ("a larger single reference supercell or a shorter
cutoff").

`cutoff=None`, which used to *prevent* aliasing by shrinking the radius to just inside the
measured boundary, has been removed in favour of this enforcement, which is stronger: it is
exact (integer ranks, no tolerance), it covers every consumer, and it reports instead of
silently folding interactions.  Callers now pass a distance or a negative shell index, and the
tutorials record the radius their reference resolves as an explicit number taken from their own
committed metadata.  What the check does not cover: whether the radius is physically right, and
the mapping tolerance that `StructureRelation` itself uses.

### Why the ASR coefficient guard stays

A first attempt replaced the de-duplication key by the integer image of each row and dropped
the coefficient guard.  Both parts failed measurement: the Cartesian row of an equation is
$e_c^\top M\,(S_{\text{lat}}B)$ while the integer row built from the same component index is
$e_c^\top (S_{\text{lat}}B)$, and these are not images of each other, so the key compared a
different object and could have merged distinct constraints; without the guard the rows of one
physical constraint no longer share a support, de-duplication keeps 13 of 18 rows instead of 1,
and the threshold-based rank test downstream then moved the constrained fit by a factor of two.
The guard itself is not sharp: it sits about $300\times$ above the noise floor measured above
and ten orders of magnitude below the smallest genuine coefficient.

### Two smaller smells

`phonon/sampling/structures.py:57,59,69` compares `cutoff_frequency != 0.01`, `imaginary_tolerance != 1e-6`
and `displacement != 0.01` as sentinels for "the caller left the default", and the same cell-comparison
question appears at three different values (`1e-5`, `1e-7`, `1e-10`) in the `structure` package.
