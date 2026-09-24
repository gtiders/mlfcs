# Force-fitting API

Fitting is a force-only linear problem. `FitSystem` consumes ASE structures with already stored forces, builds the optimized sufficient normal system, and solves all configured orders together. It does not call calculators or retain the training structures.

## Build the shared model and supercell mapping

```python
import numpy as np
from ase.build import bulk
from mlfcs import PrimitiveCell, Supercell, ClusterMap, build_cluster_space

primitive_atoms = bulk("Al", "fcc", a=4.05)
primitive = PrimitiveCell.from_atoms(primitive_atoms, symprec=1e-5)
space = build_cluster_space(
    primitive,
    cutoffs={2: 4.0, 3: 3.0},
    max_body_orders={2: 2, 3: 3},
)
supercell_atoms = primitive_atoms.repeat((3, 3, 3))
supercell = Supercell.from_atoms(
    primitive, supercell_atoms, matrix=np.diag([3, 3, 3])
)
mapping = ClusterMap.build(space, supercell)
mapping.rank_info().require_full()
```

Each order gets its own cutoff and maximum body order. The explicit supercell is the data geometry, not a property of the primitive `ClusterSpace`.

## Build and solve a fit system

```python
from ase.io import read
from mlfcs import FitSystem

training = read("training.extxyz", index=":")
system = FitSystem.from_atoms(mapping, training)
parameters = system.solve()
force_constants = system.force_constants(parameters)

print(system.n_structures, system.n_parameters)
print("force RMSE:", system.rmse(parameters), "eV/Å")
force_constants.save("fit.mlfcs")
```

Every training frame must have the same atom sequence, periodic supercell, and an ASE calculator result that already contains forces. For example, extxyz files written with forces can be read by ASE and passed directly. `FitSystem.from_atoms` reads stored results without triggering a new calculation; plain NumPy force arrays and structures with no stored force property are rejected.

The compact API is:

```python
FitSystem.from_atoms(mapping, structures)
system.solve(*, rtol=1e-8, max_steps=1000)
system.force_constants(parameters)
```

The default solve uses column-scaled MINRES on the normal system. The exact scale for parameter `i` is `1 / sqrt(H[i, i])`, where `H` is the normal matrix. A zero diagonal means that the training design never observes that parameter; the solve raises `UnobservedParameterError` rather than silently regularizing it. Add structurally informative training data or revise the model.

## Reuse, merge, and inspect

`FitSystem` stores the symmetric matrix `H = A.T @ A`, right-hand side `g = A.T @ f`, force norm, and counts. This is sufficient to solve and evaluate the least-squares problem without keeping all design rows or source structures.

- Add compatible systems with `combined = system_a + system_b`. Their cluster-space fingerprints must match.
- Pickle a system for trusted local reuse. Only unpickle files from trusted sources; pickle is executable serialization.
- Inspect `system.matrix`, `system.rhs`, `system.force_norm`, `system.n_equations`, `system.n_structures`, and `system.unobserved_parameters`.
- Assess the solution with `system.residual(parameters)`, `system.rmse(parameters)`, and `system.relative_error(parameters)`.

The RMSE is in eV/Å. Force constants of order `n` have units eV/Åⁿ. Translational or rotational post-processing is separate from the fit; for example, `force_constants.enforce_asr()` returns a projected force-constant model and a report.
