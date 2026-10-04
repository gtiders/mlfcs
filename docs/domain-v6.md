# Version 6 domain and execution contracts

The original [mathematical audit](numba-integer-audit.md) and [migration study](unified-numba-backend.md) are preserved as historical evidence. This page records the subsequently approved object merger and supersedes their API examples.

## Direct construction and explicit geometry

```python
import numpy as np
from ase.build import bulk
from mlfcs import ClusterMap, ClusterSpace, FitSystem

primitive_atoms = bulk("Si", "diamond", a=5.43)
cs = ClusterSpace(primitive_atoms, cutoffs={2: 4.0, 3: 4.0})
csmap = ClusterMap(
    cs,
    primitive_atoms.repeat((3, 3, 3)),
    supercell_matrix=3 * np.eye(3, dtype=np.int64),
)
csmap.rank_info().require_full()
# system = FitSystem(csmap, evaluated_structures)
```

Orders are the keys of `cutoffs`. Each maximum body order defaults to its tensor order. Geometry validation, spglib preprocessing, stage-local integer admission, cluster enumeration and invariant parameterization finish inside `ClusterSpace` initialization. `ClusterMap` initialization validates the supercell geometry, prepares periodic quotient tables and folds orbit images.

`cs.primitive_atoms` and `csmap.supercell_atoms` return detached ASE snapshots, including atomic masses. Domain arrays are contiguous and read-only. There is no ambiguous `.atoms` field. Folded orbit atom indices are `csmap.image_atom_indices`; atom addressing uses `csmap.atom_index(lattice_site)`.

`cs.masses` exposes readonly primitive-site masses. `cs.with_masses(values)` creates a new mass assignment sharing geometry, symmetry and orbit data, without rebuilding the model space. Rebind coefficients with `ForceConstants(new_cs, model.coefficients)`; see [harmonic frequencies and masses](harmonic-api.md).

`PrimitiveCell` and `Supercell` are removed. No geometry `build`, `from_atoms`, `map_to`, `.primitive`, `.supercell`, or `.space` compatibility aliases are provided. Taylor calculator support is removed. `FitSystem` continues to aggregate evaluated training frames; it is a data ingestion operation.

## Ownership and dependency direction

```text
ASE primitive atoms
    -> ClusterSpace (owns primitive arrays, symmetry, orbits)
        -> PrimitiveSymmetry (array representation, no ASE object)

ClusterSpace + ASE supercell atoms + integer supercell matrix
    -> ClusterMap (references ClusterSpace; owns supercell and folding arrays)
        -> ForceDesign (references ClusterMap; owns reusable compiled design)
        -> FiniteDifference (references ClusterMap)
        -> FitSystem (references ClusterSpace; owns raw equations or normal statistics)
            -> ForceConstants (references ClusterSpace; owns coefficients)
```

One `ClusterSpace` can support multiple independently initialized `ClusterMap` objects. No `MappedClusterSpace` or `ClusterModel` wrapper is introduced. Neither primitive models nor saved force constants own one mandatory supercell or retain training snapshots.

## Semantic module layout

| Module | Responsibility | Execution |
|---|---|---|
| `_arrays`, `errors` | Admission, contiguous immutable arrays, error types | Python boundary |
| `algebra/integer` | Small exact products, lattice inverses, admitted arithmetic | Python admission + Numba loops |
| `algebra/exact` | Certified exact rank and saturated kernel facade | Python orchestration |
| `algebra/_modular` | Word-bounded modular elimination and rational charts | Numba |
| `algebra/_signed` | Signed union-find kernel shortcut | Numba |
| `algebra/_congruence` | Congruence preimage and saturation | Numba |
| `core/structure`, `symmetry` | Validated arrays, coordinate conventions, spglib actions | Python preprocessing |
| `core/tensors` | Shared tensor action and Cartesian contraction | Numba + NumPy |
| `cluster_space/candidates`, `_orbits`, `basis` | Neighbors, candidates, orbit actions, invariant basis | Numba loops under owning caller |
| `mapping/geometry`, `_rank`, `cluster_map` | Explicit supercell matching, quotient addressing, exact folded rank | Python admission + Numba loops |
| `fitting/design`, `system`, `solve` | Reusable force design and per-thread temporary arrays | Python preparation + parallel Numba accumulation |
| `force_constants`, `finite_difference` | Coefficients, constraints, exports and force sampling | Python orchestration |

There is no technology-named `_numba` package or dispatch proxy. Shared exact algebra remains shared because invariant construction and folding both consume it. Folded rank calls `exact_rank`; it does not construct an integer kernel. The modular chart/congruence algorithm is unchanged by this object refactor.

## Admission, array ABI and workspace

Every compiled entry consumes contiguous `int64`/`float64` arrays and scalar metadata. NumPy allocation extents and byte lengths are checked against `np.intp`; mathematical integers use the symmetric int64 domain. Admission checks execute before the operation that needs them; domain objects do not retain or transmit certificate records.

Local proofs precede each operation on its actual data. Whole-order parameter, image and tensor-storage bounds have been removed, together with their admission branches. Sizing passes stop before exceeding actual output capacity; fill and tensor loops use admitted data without per-operation overflow checks. Periodic quotient addressing instead uses one checked Numba operation for actual additions, products and partial sums, avoiding aggregate screening and a separate fallback. Input validation, instance-specific modular reconstruction and congruence admission, exact certificates, shape checks and workspace thread-count checks still apply. Passing the geometry proof does not imply that every arbitrary exact matrix is admitted.

Numerical consumers use immutable domain arrays directly. Force designs own transformed image bases; they are derived data. Workspaces belong to a design invocation, have explicit thread capacities and must not be used concurrently. Parallel accumulation uses independent per-thread arrays followed by reduction. JIT specialization follows array dtype/layout; tensor order stays a runtime value.

## Native model storage

Version 6 writes only native format version 3: a versioned header followed by explicit model ndarray state and coefficients. Old headers are rejected before unpickling. Files remain trusted pickle artifacts, not a safe interchange format for untrusted input.

Loading normalizes arrays, validates the model layout, validates truncation inputs and checks the physical model fingerprint. It does not repeat neighbor or orbit enumeration. JIT caches, workspaces and mapping caches are not serialized. Local teaching models were converted through neutral ndarray snapshots and retained identical fingerprints. The legacy version-1 test artifact remains only for rejection coverage.

## Validation

Frozen constraint matrices, lattice oracles, orbit/action records, original Python force-design snapshots, multi-supercell folding tests and teaching workflows remain the numerical references. New tests enforce snapshot isolation, mass preservation, explicit geometry names, read-only restoration and removal of the old public types and constructors. Teaching fits keep their own complete `fit.log` in their task directory.

See [Local integer contracts](local-integer-contracts.md) for proofs, ownership and cleanup rules.
