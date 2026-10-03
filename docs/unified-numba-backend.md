# Unified Numba backend

The mathematical baseline is [Numba migration audit](numba-integer-audit.md). That
report is preserved. This document describes the implementation contract.

## Domain and dependency boundaries

`ClusterSpace` is the aggregate for a primitive interaction model. `PrimitiveCell`
and `PrimitiveSymmetry` retain their validation and read-only arrays. `Supercell`
owns explicit atoms and a periodic quotient. `ClusterMap`, in `mlfcs.mapping`, is
the derived relationship between one space and one supercell. One space can have
multiple maps; no map or training state is stored in the space.

```text
ASE/spglib → core domain objects → ClusterSpace
                                      │
                     Supercell ────────┤
                                      ▼
                                  ClusterMap
                         ┌────────────┼─────────────┐
                         ▼            ▼             ▼
                    ForceDesign  FiniteDifference  TaylorCalculator
                         ▼            ▼
                     FitSystem → ForceConstants
```

Python orchestrates these objects and NumPy/SciPy linear algebra. `backend` prepares
arrays and admission proofs. `_numba.cluster`, `_numba.exact`, `_numba.mapping` and
`_numba.design` implement reusable kernels. They import no domain objects.

## Array and safety contract

Mathematical integers, labels, permutations, indices and offsets use C-contiguous
`int64`; floating arrays use `float64`; masks use `uint8`. The admitted integer
interval excludes `INT64_MIN`, so absolute value and sign normalization are safe.
Shapes and byte lengths are checked against `np.intp` before allocation. Inputs
are read-only; writable caller arrays are copied during normalization.

The geometry entry first bounds counters by the cutoff/inverse-cell translation
box. Its neighbor counts give the actual maximum $M$, which tightens the candidate
bound $N\binom{M+p-2}{p-1}$. Subsequent label, tensor, orbit image, parameter and
allocation bounds are evaluated with Python integers. These integers describe
proofs and certificates; they do not perform characteristic-zero elimination.
`BackendCertificate` carries the result into construction. Exact chart and mapping
stages have their own admission proofs as their data becomes known.

Compiled arithmetic has no per-operation overflow helpers. Staged denominator
and numerator checks precede their multiplications. Rank/reconstruction/residual
certificates remain part of the algorithm.

`space.prepare()` returns `PreparedClusterSpace`; `mapping.prepare()` returns
`PreparedClusterMap`. A prepared map can share a caller-provided prepared space.
Compatible immutable arrays share storage. Quotient lookups are sorted arrays
with compiled binary search. There is no implicit global mapping cache.

## Exact kernel implementation

`exact_kernel(A)` returns an `int64` saturated basis and a `KernelCertificate`.
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
space = ClusterSpace.from_atoms(
    primitive_atoms, symprec=1e-5,
    cutoffs={2: 4.0, 3: 3.0}, max_body_orders={2: 2, 3: 3},
)
supercell = Supercell.from_atoms(space.primitive, supercell_atoms, matrix=matrix)
mapping = space.map_to(supercell)
system = FitSystem.from_atoms(mapping, structures)
model = system.force_constants(system.solve())
```

`ClusterSpace.build(primitive, ...)` reuses validated domain objects and optionally
symmetry. These class methods replace the public free function in version 5.
`ClusterMap` is exported from `mlfcs.mapping` and the package root; `mlfcs.supercell`
exports `Supercell`. Domain types remain separate. No `MappedClusterSpace` or
`ClusterModel` wrapper is needed because a map already references both inputs.

Native force-constant files use envelope version 2. Version 1 loads are normalized
into the current immutable/int64 domain. Models preserve their stored physical
parameterization when loaded. Prepared buffers, workspaces and construction
certificates are rebuilt rather than saved in the model format.

`ForceDesign.allocate_workspace()` returns caller-owned scratch. Streaming fitting
reuses it across snapshots. Concurrent operations must use separate workspaces.
Changing to more Numba threads than a workspace admits raises an error. Taylor
calculators share compiled code and supply material arrays at runtime, avoiding
compilation per model. Tensor order is a runtime argument; only force-design orbit
blocks currently use `prange`, with disjoint parameter columns.

## Validation and measurement

Tests compare original Python candidate labels/order, tensor contractions, and
SymPy saturated lattices; optional Rust tests compare representatives, images,
actions, integer lattices, Cartesian subspaces and folded rank. The 165 recorded
constraint fixtures cover FC2–FC6, seven crystal systems, Si/Mg, tutorial materials
and sheared cells. Exact residual, rank and Smith saturation checks cover every
fixture. Production code imports neither SymPy nor Rust; SymPy is in the reference
dependency group, and native binaries are excluded from wheels.

`benchmarks/numba_backend.py` launches isolated processes with empty Numba cache
directories and records cold/warm build and design timings, one/four threads,
workspace bytes and process peak RSS (which includes the compiler). It can compare
an original Python checkout with `--reference-source`. Results are in
`benchmarks/numba_backend_results.json`; they are workload-specific, not general
performance guarantees.

On the recorded Si FC2/FC3 workload, the warm single-thread build changed from
about 130 ms to 34 ms (3.8 times faster). Empty-cache construction changed from
2.66 s to 14.13 s; compiler-inclusive peak RSS changed from about 271 MiB to
365 MiB. Warm single-thread force-design time stayed near 7 ms. These measurements
separate compilation cost from repeated execution; they do not establish a memory
or speed improvement for every material.
