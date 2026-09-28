# Force fitting

Force fitting estimates the linear force-constant parameters from ASE structures with stored forces. A `FitSystem` represents the reusable least-squares problem for a `ClusterSpace`; all configured force-constant orders are solved together.

## `FitSystem`: streamed fitting problem

```python
FitSystem.from_atoms(mapping, structures)
system.solve(*, rtol=1e-8, max_steps=1000)
system.force_constants(parameters)
```

| Parameter | Meaning |
| --- | --- |
| `mapping` | `ClusterMap` defining the primitive model in the training supercell. |
| `structures` | Iterable of ASE `Atoms` with stored forces. Every frame must use the mapping's atom order, cell, and periodic geometry. |
| `rtol` | Relative convergence tolerance passed to the default column-scaled MINRES solve; default `1e-8`. |
| `max_steps` | Maximum MINRES iterations; default `1000`. |
| `parameters` | Complete packed parameter vector in the `ClusterSpace` order/block layout. |

The fit constructor streams structures into the compact normal system and does not retain the input frames. It only reads forces already stored on each ASE structure; calculator evaluation belongs to the data-preparation workflow.

The lower-level value constructor is `FitSystem(space, matrix, rhs, force_norm, n_equations, n_structures)`. It is useful when restoring or assembling a previously computed normal system; `space` identifies the parameter layout, `matrix` and `rhs` are $A^T A$ and $A^T f$, `force_norm` is $f^T f$, and the final two integers record the equation and structure counts. Most workflows should use `FitSystem.from_atoms(mapping, structures)` so those quantities are accumulated from validated force-bearing structures.

## Minimal Si fitting example

```python
from ase.io import iread, read
from mlfcs import ClusterMap, ClusterSpace, FitSystem, Supercell

space = ClusterSpace(
    read("POSCAR"),
    cutoffs={2: -7, 3: -6, 4: -3, 5: -2},
    max_body_orders={2: 2, 3: 3, 4: 4, 5: 3},
)
supercell = Supercell.from_atoms(space.primitive, read("SPOSCAR"))
mapping = ClusterMap.build(space, supercell)
mapping.rank_info().require_full()

system = FitSystem.from_atoms(mapping, iread("train.xyz", index=":"))
parameters = system.solve(rtol=1e-8, max_steps=10_000)
model = system.force_constants(parameters)
model.save("fc-si-fit.mlfcs")
```

This mirrors the repository's [Si fitting script](../tutorial/SI/fitting/fit.py), which additionally applies ASR and exports FC2 and FC3. Training forces may come from DFT or a machine-learning potential; once stored on ASE `Atoms`, the fitting interface is the same.

## The linear system and solver parameters

For design matrix $A$, force vector $f$, and packed parameters $p$, the fitting objective is $min_p \|Ap-f\|_2$. The default path accumulates $H=A^T A$ and $g=A^T f$, then solves the column-scaled normal system with MINRES. The scale for column $i$ is $1/\sqrt{H_{ii}}$.

The fit exposes `matrix`, `rhs`, `force_norm`, `n_equations`, `n_structures`, `n_parameters`, `unobserved_parameters`, and `fingerprint`. `residual(parameters)`, `rmse(parameters)`, and `relative_error(parameters)` evaluate the force residual represented by the system. `parameter_name(index)` maps a packed parameter index to its order, orbit, and component.

If a diagonal element $H_{ii}$ is exactly zero, that parameter is absent from the training design. `solve()` raises `UnobservedParameterError`; add informative structures or change the model. `rtol` controls iterative convergence, while `max_steps` limits solver iterations.

## Reuse and combine fit systems

`FitSystem` stores the normal matrix, right-hand side, force norm, and counts. Compatible systems can be added with `system_a + system_b`; their cluster-space identities must agree. Fit systems are pickleable for trusted local reuse. Pickle files can execute code when loaded, so only load files from trusted sources.

## Retain the original equations with `FitData`

`FitData` is the opt-in representation for workflows that need the original design rows, for example to call a different least-squares solver:

```python
import numpy as np
from mlfcs import FitData

data = FitData.from_atoms(mapping, iread("train.xyz", index=":"))
design, forces = data.arrays()
parameters = np.linalg.lstsq(design, forces, rcond=None)[0]
model = data.force_constants(parameters)
```

Its direct value constructor is `FitData(space, designs, forces)`: `designs` and `forces` each contain one matrix/vector pair per structure. Public methods are `from_atoms(mapping, structures)`, `arrays()`, `normal_system()`, and `force_constants(parameters)`. `designs` and `forces` provide per-structure arrays; `n_structures`, `n_equations`, and `n_parameters` report their dimensions. `FitData` retains the full design and therefore uses memory proportional to the number of force equations. `FitSystem.from_atoms` is the compact streamed route.

The fitted model can be projected and exported through [force constants](force-constants.md).
