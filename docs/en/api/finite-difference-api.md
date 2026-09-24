# Finite-difference API

`FiniteDifference` generates a deterministic, order-defined sequence of ASE structures. The sequence is also the data contract for reconstruction: structure `i` returned by `displacements()` must be paired with the forces calculated for that exact geometry and atom order.

## Define the primitive model and supercell

```python
import numpy as np
from ase.build import bulk
from mlfcs import (
    PrimitiveCell, Supercell, ClusterMap, build_cluster_space, FiniteDifference,
)

primitive_atoms = bulk("Al", "fcc", a=4.05)
primitive = PrimitiveCell.from_atoms(primitive_atoms, symprec=1e-5)
space = build_cluster_space(
    primitive,
    cutoffs={2: 4.0},
    max_body_orders={2: 2},
)
supercell_atoms = primitive_atoms.repeat((3, 3, 3))
supercell = Supercell.from_atoms(
    primitive, supercell_atoms, matrix=np.diag([3, 3, 3])
)
mapping = ClusterMap.build(space, supercell)
mapping.rank_info(2).require_full()
```

`symprec` is the declared geometric tolerance in Å used for symmetry and primitive-to-supercell mapping. The integer supercell matrix is mandatory. The caller supplies the supercell and is responsible for selecting one large enough to identify the requested interactions.

## Generate and evaluate displacements

```python
from ase.calculators.emt import EMT

fd = FiniteDifference(mapping, order=2, disps=(0.01, 0.02))
displaced = fd.displacements()
evaluated = fd.evaluate(EMT())
fc2 = fd.reconstruct(evaluated)
```

The example uses ASE's EMT as a compact calculator demonstration. Any ASE calculator can be substituted if it supports the elements and structures. `evaluate(calculator)` explicitly forces a fresh force calculation for every displacement and stores each result on the returned `Atoms` with a single-point calculator.

The constructor is:

```python
FiniteDifference(mapping, *, order: int, disps: float | Sequence[float] = 0.01)
```

- `order` is the force-constant order and must be at least 2.
- `disps` is one positive displacement length in Å or a sequence of distinct positive lengths. Multiple lengths are extrapolated to zero displacement using the even-error polynomial in displacement squared.
- The selected order must exist in the cluster space, and the supercell mapping must be structurally identifiable for that order.

## External calculators and persisted structures

An external code may evaluate the structures instead of `fd.evaluate`. Write the output of `fd.displacements()` to a format that preserves atom order and cell, run the external calculation, then read the structures back in the same order. Attach each force array to its corresponding ASE object:

```python
from ase.calculators.singlepoint import SinglePointCalculator

for atoms, forces in zip(displaced, force_arrays, strict=True):
    atoms.calc = SinglePointCalculator(atoms, forces=forces)

fc2 = fd.reconstruct(displaced)
```

`force_arrays` here means the forces read from the external calculation, one array per structure with shape `(n_atoms, 3)`. It is not itself accepted by `reconstruct`: the ASE structures carry the required geometry, atom ordering, and force association. Use an ordered format such as extxyz, and do not sort, deduplicate, or otherwise reorder frames.

## Reconstruction contract and errors

`fd.reconstruct(structures)` accepts only an ordered sequence of ASE `Atoms` with stored forces. It checks the sequence length, atomic numbers and order, periodicity, cell and periodic positions against the generated structures, then validates finite force arrays. Reconstruction refuses mismatches rather than guessing a permutation.

Common failures:

- `AliasingError`: the explicit supercell cannot distinguish all requested primitive parameters. Change the supercell or reduce the model cutoff/body order.
- Missing stored forces: attach the force results to each ASE structure; reconstruction does not run a calculator.
- Geometry or order mismatch: restore the original frame order and atom ordering before reconstruction.

The result is a primitive `ForceConstants` object containing only the selected order. It can be combined with disjoint orders using `ForceConstants.combine`, saved in the native format with `fc2.save("fc2.mlfcs")`, or exported through `fc2.write(...)`.
