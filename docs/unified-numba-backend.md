# Unified Numba backend

The mathematical baseline is [Numba migration audit](numba-integer-audit.md). That
report is preserved. This document describes the implementation contract.

## Domain and dependency boundaries

`ClusterSpace` owns the immutable primitive interaction model and
`PrimitiveSymmetry`. `ClusterMap` owns one explicit supercell realization and
the derived quotient, atom and folded-cluster mappings. A space can have many
maps; neither the space nor its maps retain training structures.

```text
ASE/spglib → ClusterSpace ← ASE supercell atoms
                    │                │
                    └──────→ ClusterMap
                               ├── ForceDesign → FitSystem → ForceConstants
                               ├── FiniteDifference
                               └── reciprocal calculations
```

Python owns validation, spglib calls, orchestration and NumPy/SciPy linear algebra.
Numba kernels live beside the domain operation that owns their inputs; they take
contiguous arrays and scalar metadata, not ASE objects or Python containers. Shared
periodic geometry lives in `core/geometry.py` and is evaluated in Cartesian coordinates
after Minkowski reduction. Site matching enumerates images within the declared tolerance
once; fitting displacements use the separate nearest-image query.

## Array and safety contract

Mathematical integers, labels, permutations, indices and offsets use C-contiguous
`int64`; floating arrays use `float64`; masks use `uint8`. The admitted integer
interval excludes `INT64_MIN`, so absolute value and sign normalization are safe.
Shapes and byte lengths are checked against `np.intp` before allocation. Inputs
are read-only; writable caller arrays are copied during normalization.

The geometry entry first bounds counters by the cutoff/inverse-cell translation
box. Its neighbor counts give the actual maximum $M$, which tightens the candidate
bound $N\binom{M+p-2}{p-1}$. Subsequent label, tensor, orbit image, parameter and
allocation bounds are evaluated with Python integers at the stage where those
values are known. These integers describe local bounds; they do not perform
characteristic-zero elimination or travel with domain objects.

Hot loops use int64 arithmetic. Periodic quotient mapping checks each actual
addition, product and partial sum in its single Numba mapping pass. Staged
denominator and numerator checks precede their multiplications. Rank,
reconstruction and residual certificates remain part of the exact algorithm.

There are no `PreparedClusterSpace` or `PreparedClusterMap` wrappers. Immutable
domain arrays are passed directly to numerical consumers. Quotient lookups are
owned by `ClusterMap` as sorted arrays with compiled binary search. There is no
implicit global mapping cache.

## Exact kernel implementation

`exact_kernel(A)` returns an `int64` saturated basis.
Signed incidence constraints take the signed union-find path. General matrices
take a fixed two-prime pivot chart, rational reconstruction, and a composite
congruence preimage. Independent modular annihilation certificates, with residual
bounds, establish the reconstructed chart's exact upper rank; the nonzero pivot
minor establishes its lower rank. Folded rank uses certified rank directly.

The composite step is a specialized triangular preimage algorithm, rather than a
general-purpose Howell library or finite-field RREF over a composite modulus.
It produces column HNF: positive diagonal, upper triangular, and
$0\le H_{ij}<H_{ii}$ for $i<j$.

For one row $w$, put $g_{-1}=\delta$ and
$g_j=\gcd(\delta,w_0,\ldots,w_j)$. The $j$th diagonal pivot is
$g_{j-1}/g_j$. A bounded Bezout vector represents $g_{j-1}$ modulo $\delta$;
it supplies the preceding coordinates of that column. All coordinates are
reduced modulo $\delta$, then by preceding columns. The constructed columns
satisfy the congruence and their determinant is $\delta/g_{d-1}$, exactly the
index of the congruence kernel, so they generate the full preimage.

For multiple rows, intersect the current $H\mathbb Z^d$ with each congruence by
computing the one-row preimage $T$ of $w=F_iH\bmod\delta$. The new lattice has basis
$HT$. It still contains $\delta\mathbb Z^d$, so its triangular diagonal divides
$\delta$. Off-diagonal multiplication can be performed modulo $\delta$ because
subtracting $\delta e_i$ belongs to the preceding column lattice. Reducing by
already constructed columns yields the canonical column HNF. Stored entries are
at most $\delta$, each product is below $\delta^2$, and each modular sum/subtraction
is reduced immediately. No full unimodular transform is maintained.

The lift $(-FH/\delta;H)$ uses quotient/remainder accumulation with the admitted
bound $d(\max|F|+2\delta+1)$. Rational reconstruction uses alternating convergent
coefficient signs and the Euclidean determinant invariant to bound coefficient
products independently of remainder products. LCM and numerator scaling are
checked before multiplication.

The current reconstruction domain uses primes 2147483647 and 2147483629,
denominator windows through $2^{30}$, common denominator below $2^{31}$, and the
accumulator bound above. A singular fixed pivot chart or failed reconstruction is
reported; there is no bigint or floating fallback. This is a sufficient dynamic
admission domain, not support for every arbitrary-precision integer matrix.

## Public API and persistence

```python
space = ClusterSpace(
    primitive_atoms, symprec=1e-5,
    cutoffs={2: 4.0, 3: 3.0}, max_body_orders={2: 2, 3: 3},
)
mapping = ClusterMap(space, supercell_atoms)
system = FitSystem(mapping, structures)
model = system.solve()
```

`ClusterSpace` accepts ASE primitive atoms directly. `ClusterMap` infers the
supercell matrix from its ASE atoms when no matrix is supplied. `ClusterMap` is
exported from `mlfcs.mapping` and the package root. `PrimitiveCell`, `Supercell`,
Taylor calculators and the old `prepare()` APIs are not part of the current API.

Native force-constant files use HDF5 format version 4; older native files are
rejected. Explicit arrays preserve the stored physical parameterization and masses
without reconstructing orbits. Only ForceConstants offers save/load; workspaces,
JIT caches and object graphs are not serialized. Callers own compatibility of
models and mappings; no model identity hashes are computed or compared.

`ForceDesign.allocate_workspace()` returns caller-owned scratch. Streaming fitting
reuses it across snapshots. Concurrent operations must use separate workspaces.
Changing to more Numba threads than a workspace admits raises an error. Taylor
calculators share compiled code and supply material arrays at runtime, avoiding
compilation per model. Tensor order is a runtime argument; only force-design orbit
blocks currently use `prange`, with disjoint parameter columns.

## Validation and measurement

Tests compare original Python candidate labels/order, tensor contractions and
SymPy saturated lattices. Frozen symmetry and force-design fixtures cover multiple
orders, structures and sheared cells. Exact residual, rank and saturation checks
guard the exact algebra path. Production code imports neither SymPy nor Rust;
SymPy is in the reference dependency group, and native binaries are excluded from
wheels.

Performance measurements are workload-specific. Always report cold JIT separately
from warmed execution, record thread count and peak memory, and compare complete
construction/design paths as well as individual kernels. A microbenchmark does not
establish performance for every material.
