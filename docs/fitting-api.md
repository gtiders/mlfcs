# Force-fitting API

`FitSystem` owns one force-only least-squares problem. Initialize it directly with a `ClusterMap` and evaluated ASE structures. `representation="normal"` is the default and uses MINRES; `representation="raw"` retains the original equations and uses LSMR. These are the only built-in algorithms. `solve()` returns `ForceConstants`.

## Initialize and solve

```python
from ase.io import iread, read
from mlfcs import ClusterMap, ClusterSpace, FitSystem

space = ClusterSpace(
    read("POSCAR"),
    cutoffs={2: 4.0, 3: 3.0},
    max_body_orders={2: 2, 3: 3},
)
mapping = ClusterMap(space, read("SPOSCAR"))
system = FitSystem(mapping, iread("train.extxyz", index=":"))
model = system.solve(rtol=1e-8, maxiter=1000)
print(system.rmse(model), "eV/Å")
model.save("fit.mlfcs")
```

The integer supercell matrix may be supplied to `ClusterMap`; otherwise it is inferred. `FitSystem` checks structural identifiability when preparing its internal force design. Each training frame must have the reference atom sequence, periodicity and cell, and already stored finite ASE forces. Construction never invokes a calculator. A frame generator is consumed once; create another iterator when constructing a second system.

## What happens to the data

For each frame, positions are compared with the reference supercell using Cartesian minimum periodic images. The resulting displacement $u_i$ enters the symmetry-reduced Taylor force design $A_i$. The force array is flattened in atom order, with Cartesian x/y/z components consecutive. Columns follow `cluster_space.parameter_offsets` and the cluster space's order/orbit/component layout. The Taylor signs, factorials, tensor bases and image sums are already incorporated into $A_i$.

No mean-force subtraction, per-frame weighting or data normalization is applied during construction. Both representations describe physical parameters $\theta$ and the same problem:

$$
\min_\theta \|A\theta-f\|_2^2.
$$

| Representation | Public data | Algorithm | Storage |
|---|---|---|---|
| `raw` | `design_matrix` ($A$), `forces` ($f$) | LSMR | $O(mn)$ |
| `normal` | `normal_matrix` ($H$), `normal_rhs` ($g$) | MINRES | $O(n^2)$ |

Here $m$ is the total number of Cartesian force equations and $n$ is the number of parameters. Raw frames are stacked in iteration order. The normal route accumulates, frame by frame,

$$
H=\sum_i A_i^T A_i,\qquad g=\sum_i A_i^T f_i,\qquad c=\sum_i f_i^T f_i.
$$

`force_squared_norm` stores $c$ in either representation. `n_structures`, `n_equations`, `n_parameters`, `representation` and `cluster_space` describe the system. Public arrays are readonly, retain physical scale, and remain unchanged by solving. Accessing an array belonging to the other representation raises `ValueError`. Normal systems discard the raw rows and cannot recover individual-frame residuals.

Raw construction temporarily retains frame blocks while stacking them, and raw solving allocates a normalized working matrix. Plan for temporary storage in addition to the final $mn$ matrix. A normal system streams frames without retaining their design matrices, but its matrix can still be large. For full-rank $A$, $\kappa_2(A^TA)=\kappa_2(A)^2$; accumulating normal equations can lose information in poorly conditioned problems.

## Normalization belongs to solving

`fitting/solve.py` implements the internal `FitSolver` class used by `system.solve()`. It always normalizes parameter columns; the representation fixes the algorithm.

For raw equations, let

$$
s_j=\frac{1}{\|A_{:j}\|_2},\qquad S=\operatorname{diag}(s_j).
$$

LSMR solves $ASz\approx f$, and the returned physical parameters are $\theta=Sz$. The force vector is not rescaled. Column maxima are used before computing norms to avoid squaring very large or very small physical coefficients. The normalized matrix is temporary, so user-accessible $A$ and $f$ keep their original values.

For normal equations, $s_j=1/\sqrt{H_{jj}}$. MINRES solves

$$
SHSz=Sg,\qquad \theta=Sz.
$$

This is the existing column-scaled iteration. Scaling balances parameter columns from different FC orders; it cannot resolve missing observations or linear dependence. Exact geometric folded rank and the numerical rank of the training equations are distinct. A zero column raises `UnobservedParameterError`. In a rank-deficient problem the solver operates in scaled coordinates, so a selected solution need not minimize the norm of physical $\theta$.

| Route | Defaults | Meaning |
|---|---|---|
| normal / MINRES | `rtol=1e-8`, `maxiter=1000` | MINRES relative stopping tolerance and iteration limit |
| raw / LSMR | `atol=1e-8`, `btol=1e-8`, `conlim=1e8`, `maxiter=1000` | Matrix/least-squares tolerance, force-vector tolerance, scaled condition limit, iteration limit |

LSMR's compatible-system stopping test uses approximately $\|r\|\le\texttt{atol}\|AS\|\|z\|+\texttt{btol}\|f\|$; its least-squares test controls $\|(AS)^Tr\|$. `conlim=0` disables the condition-limit test. Tolerances are dimensionless and apply to the normalized problem. A convergence failure, iteration-limit stop or condition-limit stop raises an error. Options for the other algorithm, `solver=` and additional algorithms are not supported. See [SciPy MINRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.minres.html) and [SciPy LSMR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html) for the stopping rules.

```python
raw = FitSystem(mapping, iread("train.extxyz", index=":"), representation="raw")
model = raw.solve(atol=1e-10, btol=1e-10, conlim=1e8, maxiter=2000)
```

## External solvers

External solvers receive physical equations and control their own normalization. For example, call SciPy LSMR directly on the public raw arrays:

```python
from scipy.sparse.linalg import lsmr

A, f = raw.design_matrix, raw.forces
result = lsmr(A, f, atol=1e-10, btol=1e-10, maxiter=2000)
if result[1] not in (0, 1, 2, 4, 5):
    raise RuntimeError(f"external LSMR stopped with code {result[1]}")
parameters = result[0]
model = raw.force_constants(parameters)
print(raw.rmse(model))
```

Unlike `raw.solve()`, this direct call does not apply MLFCS column normalization. Inspect the external solver's stop status before accepting its solution. If you normalize externally, solve with $AS$ and pass $Sz$ to `force_constants()`, never the scaled coordinates $z$. The parameter vector must be finite and have length `n_parameters` in the original canonical layout. Force parameters of order $p$ have units eV/Å$^p$; displacement input is in Å and forces are in eV/Å.

For a normal system, give `system.normal_matrix` and `system.normal_rhs` to your symmetric-system solver, then bind its physical result with `system.force_constants(parameters)`. The saved matrix is $H$, not $A$. Applying least-squares to $(H,g)$ changes the residual objective and must not be described as using the original force equations.

## Conversion, merging and diagnostics

- `raw.to_normal()` returns another `FitSystem` with compressed physical equations. A normal system's `to_normal()` returns itself. There is no reverse conversion.
- `a + b` adds compatible normal statistics or stacks compatible raw equations. Representations must match; callers must ensure that physical parameter layouts match. Model geometry and basis identity are not checked. Explicitly convert raw systems before merging with normal systems.
- `residual(model_or_parameters)`, `rmse(model_or_parameters)` and `relative_error(model_or_parameters)` evaluate physical force errors. Raw residuals are evaluated directly; normal residuals use $\theta^TH\theta-2\theta^Tg+c$ and are subject to cancellation near a perfect fit.
- FitSystem has no supported persistence API. Save the fitted ForceConstants with save(); callers may store the exposed equation arrays themselves when needed.
- ASR and rotation projection remain explicit operations on the resulting `ForceConstants`.

`FitData`, `FitSystem.from_atoms()`, `matrix`, `rhs`, `force_norm`, `column_scale` and `max_steps` are removed. Use direct construction, explicit data properties and `maxiter`. `solve()` now returns a model; obtain its physical vector with `model.parameters()`.
