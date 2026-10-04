# Local integer contracts and cleanup rules

This document describes the current implementation. The existing mathematical audit and its G chapter are historical baselines and have not been edited.

## Policy

A local proof replaces a whole-problem admission bound when it checks the same operation at the point where its actual operands are known. Remove the redundant whole-problem calculation, carried certificate fields, branches and tests. Use one checking algorithm for each operation; do not retain an aggregate screen and a separate local fallback for the same arithmetic.

A local bound still bounds intermediates conservatively; failure means the operation cannot be certified under that contract, not that its final mathematical answer necessarily exceeds int64.

Python integers are used for boundary bounds and small lattice preprocessing. Production elimination, reconstruction, congruence and tensor kernels use int64; there is no bigint elimination fallback.

Separate these obligations:

- arithmetic: products, partial sums, differences, negation and indices;
- allocation: actual extents and byte lengths;
- schema: shapes, declared integer dtype, permutations and valid indices;
- exactness: rank, annihilation, reconstruction and saturation;
- numerical input and physical constraints: finite values, geometry tolerance and fitting observations.

An arithmetic proof does not replace the other obligations.

Admission checks belong to the operation that needs them. Domain objects do not carry proof records. Exact kernel returns only the readonly saturated basis; rank and annihilation certification still execute within the operation. Mapping `RankInfo` remains a business result rather than an admission token.

## Arithmetic and allocation conventions

Let $J=2^{63}-1$ and $J_p=\mathrm{np.iinfo(np.intp).max}$. Integer normalization admits the symmetric domain $[-J,J]$, excluding the minimum int64 value so absolute values and negation remain safe.

For an actual array with shape $(s_1,\ldots,s_k)$ and item size $w$, require nonnegative extents representable by `np.intp` and

$$
w\prod_i s_i\le J_p.
$$

This is a representation bound, not an available-RAM guarantee. A representable allocation can still raise `MemoryError`. Checks precede allocation and fixed-width conversion.

## Geometry sizing

`cluster_space/candidates.py` validates the actual geometry and truncation inputs, computes traversal bounds and counts neighbors immediately before allocation and fill. Bounds and offsets are local algorithm data; no admission record is returned or stored on the model.

Each neighbor traversal checks $2b_j+1\le J$, covering `range(-b_j,b_j+1)` endpoints and length. The sizing kernel stops before its counter exceeds the representable capacity of the actual neighbor-label array. A negative status rejects the sizing result before allocation or prefix conversion.

Candidate sizing similarly caps its counter at the capacity of the actual $(C,p,4)$ labels. Both kernels specialize sizing and fill separately; the admitted fill path omits counter capacity checks. Repetition and candidate ordering are unchanged.

Before traversal, check geometry, order, cutoff and body order. The count pass writes offsets directly; fill reuses those offsets on the same readonly inputs. No count tuple, prefix-sum reconstruction or certificate matching is needed.

## Orbit actions

For actual cluster translation coordinate maxima $T_k$, operation $g$ and consumed sites, define

$$
E_{g,j}=\max_s|h_{g,s,j}|+\sum_k|R_{g,jk}|T_k.
$$

Absolute products and partial sums are bounded by $E_{g,j}$. Reanchoring subtracts two admitted results, so $2\max_{g,j}E_{g,j}\le J$ suffices for the whole action. Primitive site queries use the selected operation and site without the reanchoring factor.

Actual action output, deduplication capacity, and registry initial/grown capacity have separate allocation checks. Normalized symmetry arrays do not bypass the action proof. Registry hashing has fixed small modular arithmetic bounds.

## Tensor actions and invariant basis

For actual rotation $R$, order $p$, input coefficient maximum $V$ and

$$
L=\max\left(1,\max_j\sum_k|R_{jk}|\right),
$$

the axis-$t$ values, products and partial sums are bounded by $VL^t$. Thus $VL^p$ covers every stage. Label-symmetric seed coefficients have $V=1$; subtracting the seed adds one to the bound.

Each actual stabilizer is certified before constructing constraints. Allocation checks protect the real label basis, tensor frame, permutations and $(H D_p,k)$ constraint array. No hypothetical $(C_p,D_p,D_p)$ tensor array is admitted.

Parameter offsets are built from actual orbit dimensions and validated before conversion to int64.

## Exact algebra

The signed path first constructs its component metadata, determines nullity $d$, validates the actual $(n,d)$ basis allocation, then fills it. The modular path checks its actual kernel and congruence workspace shapes.

Primes and CRT constants have one source in `_modular.py`. For $q<2^{31}$, residues lie in $[0,q-1]$, giving

$$
(q-1)^2+(q-1)=q(q-1)<2^{62}.
$$

This covers modular multiplication, reduction and modular accumulation. For CRT, $0\le a<p_1$ and $0\le t<p_2$ imply

$$
a+p_1t\le p_1p_2-1<J.
$$

Rational reconstruction preserves the Euclidean alternating-sign coefficient recurrence. Its coefficient products are bounded by the next coefficient magnitude, which is at most the CRT modulus. This is a coefficient invariant, not a consequence of the remainder-product inequality.

Common-denominator construction retains its dynamic multiplication guards. With $u=\delta/\gcd(\delta,d)$, verify $u\le\lfloor\delta_{\max}/d\rfloor$ before multiplying. Verify numerator scaling before multiplying as well.

Congruence preimage uses upper-triangular column HNF with positive diagonal and $0\le H_{ij}<H_{ii}\le\delta$ for $i<j$. The lattice contains $\delta\mathbb Z^d$, so the diagonal divides $\delta$. Reduced products are bounded by $\delta^2$; lift accumulation uses the actual bound

$$
d(\max|F|+2\delta+1)\le J.
$$

Exact rank certification and modular annihilation verification remain mandatory. Their Python-integer bounds may exceed int64 because the verified product is evaluated modulo machine-word primes. They are not fixed-width dot-product admission checks.

## Integer products and mapping

For an actual integer product, require for each output entry

$$
\sum_k|A_{ik}B_{kj}|\le J.
$$

Every multiplication and partial sum then fits, even when the final result involves cancellation. There is no separate maximum-coefficient product gate.

Periodic quotient queries and label mapping share one Numba quotient implementation. Label additions, quotient products and each partial sum are checked immediately before evaluation, using the actual operands in the symmetric int64 domain. There is no aggregate screening bound, fallback or second preflight traversal. Binary search uses `left + (right-left)//2`.

ClusterMap owns one immutable periodic index reused by mapping and queries. Arbitrary translation queries use the same checked quotient kernel. Export translation differences are computed as Python integers before normalization, avoiding an implicit dependence on an unrelated old quotient bound.

## Periodic geometry

Primitive-site matching, supercell atom matching and supercell matrix inference all use one fixed-radius search. In the reduced cell $B$, every accepted image satisfies $|h_j+(vB^{-1})_j|<\epsilon\|(B^{-1})_{:j}\|$. Enumerate this integer box once and accept only Cartesian lengths below `symprec`; count every match to enforce uniqueness. There is no nearest-image prefilter or repeated search. Fitting displacements require the nearest image rather than all images below a tolerance and therefore retain their distinct minimum-image query.

## Folded rank

Distinct orbits occupy disjoint parameter columns. For a folded atom tuple $a$ and orbit $o$, use

$$
U_{a,o}=\sum_{\text{image}\in(a,o)}V_o L_{g_{\text{image}}}^{p}.
$$

This bounds all transformations contributing to that block and every prefix of its accumulation. Other orbits and unrelated atom tuples do not enlarge the bound. Actual alias-component matrix shapes are checked before allocation. The existing exclusive-image full-rank shortcut runs without preparing unnecessary folding matrices or bounds.

## Remaining algorithm selection

Removing redundant checking routes does not require deleting structural solutions:

- Signed constraints have a direct saturated component basis. This path also accepts
  scaled signed rows whose coefficients vanish at a reconstruction prime. For example,
  $[p,-p]$ has basis $(1,1)^T$, while the current general chart rejects that pivot at
  prime $p$. Removing this path without a replacement changes the admitted domain.
- An exclusive folded image proves injectivity of its orbit parameter block directly.
  Keep this proof instead of allocating and ranking redundant transformed matrices.
- Rational reconstruction tries denominator windows within the same algorithm.
  A single largest window imposes a smaller numerator limit and can reject charts
  accepted by a smaller window. These iterations and their exact verification remain.
- Neighbor counting and filling are two passes of the same enumeration, needed to
  allocate variable-size outputs. They are not alternate enumeration algorithms.

The mapping operation has one arithmetic route, and tolerance matching has one
search route. Empty-input handling, validation failures and exact certificates
remain part of those algorithms.

## Design and fitting lifecycle

ForceDesign validates actual image/basis buffers and fixed output shape during initialization. Snapshot calls retain displacement validation and mutable-workspace validation, but do not repeat fixed output-size admission.

Workspace allocation checks actual thread capacity and scratch shape. Every use checks dimensions, scratch count, dtype, layout, writability and active thread capacity. Workspaces are caller-owned and cannot be shared concurrently.

FitSystem ingestion checks actual normal or raw matrix and force-vector allocation. Solver column normalization is temporary; stored fitting data retains physical scale. Periodic indexes and workspaces are not serialized. Loading validates stored arrays, truncation inputs and model layout without neighbor enumeration. Numerical consumers use the domain object's existing read-only arrays directly.

## Consolidation and cleanup

- Neighbor/candidate admission and sizing: `cluster_space/candidates.py`.
- Invariant basis and Cartesian parameterization: `cluster_space/basis.py`.
- Structure validation and lattice addresses: `core/structure.py`.
- Supercell validation and quotient mapping: `mapping/geometry.py`.
- Design and scratch: `fitting/design.py`.
- Training ingestion and sufficient systems: `fitting/system.py`.
- Displacements and reconstruction: `finite_difference/difference.py`.
- External format adapters: `force_constants/io.py`.

Modular, composite-congruence, signed-graph and exact-facade boundaries remain separate. Shared acoustic equations remain independent of ASR and rotation. Public constructors and native model format are unchanged.

Delete an old proof only after its replacement covers the same operations and data, including all intermediate values and actual allocation shapes. Delete dead code only after checking production imports, public exports, restoration and downstream usage. Keep test oracles, historical reports, teaching logs and numerical reference data. Do not create tests for removed global bounds.

## Validation record

Validated on Python 3.14.4, NumPy 2.5.3 and Numba 0.68.0:

- 101 tests passed, including 165 recorded FC2–FC6 exact-constraint matrices, bigint oracle lattice checks, original Python design/fitting snapshots and local-contract tests.
- KAsPt was reconstructed from its saved model inputs: 333 orbits and 6849 parameters. Clusters, actions, exact/Cartesian bases, mappings, the sampled design matrix and predicted forces were bitwise identical to the saved model. The force-constant fingerprint remained `9199da2c45f7f657914acb79a68b21f86083c8f61e23deb35f349e2e173bd40a`.
- Historical audit and architecture-report SHA-256 hashes were unchanged.
- Ruff checks and formatting passed (excluding the pre-existing executable-bit warning on the cutoff utility), documentation validation and strict MkDocs build passed, and wheel/sdist packaging succeeded.
- No teaching fits or thermal-conductivity calculations were rerun during this cleanup. Ba thermal conductivity was not run.

The Si benchmark uses FC2/FC3, cutoff 4 Å, a 3×3×3 supercell and 43 parameters. Cold construction uses a separate empty JIT cache. Warm design values average 100 calls; single-thread and two-thread matrices were bitwise identical.

| Measurement | Result |
|---|---:|
| Cold construction (s) | 15.636 |
| First design call (s) | 1.088 |
| Warm construction (s) | 0.229 |
| Warm single-thread design (ms) | 7.216 |
| Warm two-thread design (ms) | 4.854 |
| Peak RSS (MiB) | 438.383 |

The earlier construction baseline was approximately 0.118 s warm and 12.954 s cold. Local proofs increase preparation cost; this single benchmark run does not establish a speedup. The earlier warm design measurements were 6.551 ms single-thread and 4.241 ms with two threads.


## Admission lifecycle validation (2026-10-03)

Removed `OrderCertificate`, `ClusterCertificate` and `KernelCertificate` rather than replacing them with another proof record. Candidate traversal owns its local preparation, exact kernel returns only the saturated basis, and loaded models validate stored data without neighbor enumeration. The native file version remains 3.

- All 100 collected tests passed, including the 165 recorded FC2–FC6 constraint matrices, lattice oracle checks, original Python force-design/fitting snapshots, invalid truncation restoration and loading with neighbor enumeration disabled.
- KAsPt's saved model loaded without neighbor search and retained 333 orbits, 6849 parameters and its existing fingerprint. Prepared positions still share the model's readonly storage.
- The two historical report hashes remained unchanged. Ruff checks and formatting passed for the changed modules/tests; the documentation checker passed. Full-source Ruff additionally reports a pre-existing slot-order issue in `tools/perturbation.py` (with the pre-existing executable-bit issue excluded). Strict MkDocs building was unavailable because MkDocs is not installed in this environment.

Ba measurements use FC4 only, cutoff 5 Å, maximum body order 4, one thread, three constructions per process and separate empty JIT caches. The comparison restored the previous certificate path in a temporary source copy; it is not a Git commit baseline. Both implementations returned 2838 orbits and 155586 parameters.

| Measurement | Restored previous contract path | Current local admission |
|---|---:|---:|
| Cold initialization (s) | 25.074 | 25.241 |
| Warm initialization, median of two (s) | 11.135 | 11.186 |
| Peak RSS across three runs (MiB) | 648.336 | 632.090 |

The paired initialization timings are essentially unchanged. The earlier historical warm measurement of 9.083 s was not reproduced by either path in this run; these samples do not establish an initialization speedup. Warm KAsPt loading took 0.259 s for the restored previous path and 0.246 s for the current implementation (medians of four calls); the stronger lifecycle regression is that current loading succeeds when neighbor enumeration is forbidden. Measurements remain in `/tmp/mlfcs-ba-fc4-lifecycle-*.json` and `/tmp/mlfcs-lifecycle-load*.json`.

No teaching fits or thermal-conductivity calculations were rerun.
