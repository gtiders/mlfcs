# Core concepts: from a primitive model to supercell data

MLFCS separates the **model** from the **structure used to obtain forces**. Learn these three objects first:

| Object | Owns | Does not own |
| --- | --- | --- |
| `ClusterSpace` (`space`, often called CS) | A primitive cell, its symmetry, clusters, orbits, and independent force-constant parameters for one or more orders | A supercell, displacements, or training forces |
| `Supercell` | An explicit ASE supercell, its inferred integer relation to the primitive cell, and the primitive site represented by each atom | Clusters or force-constant parameters |
| `ClusterMap` (`mapping`) | The connection between one CS and one supercell: which supercell atoms represent each primitive cluster image | A second set of orbits or fitted parameters |

The flow is `primitive ASE Atoms → ClusterSpace`, `explicit supercell ASE Atoms → Supercell`, then `(ClusterSpace, Supercell) → ClusterMap`. Finite difference and fitting consume the mapping; their result is a `ForceConstants` model attached to the primitive CS.

## ClusterSpace: the primitive model

```python
from ase.build import bulk
from mlfcs import ClusterSpace

primitive_atoms = bulk("Si", "diamond", a=5.43)
space = ClusterSpace(
    primitive_atoms,
    cutoffs={2: 4.0, 3: 3.0},
    max_body_orders={2: 2, 3: 3},
    symprec=1e-5,  # Å; this is also the default
)

print(space.orders, space.n_parameters)
```

`ClusterSpace` creates and keeps its own `space.primitive`. It discovers primitive symmetry once at `symprec` and uses it to group equivalent clusters into orbits. Each orbit contributes independent Cartesian tensor parameters. The example defines FC2 and FC3 together; each order has its own cutoff and maximum number of distinct atomic sites in a cluster.

A positive cutoff is a radius in Å. A negative integer, such as `-7`, requests the seventh neighbor shell; CS resolves that request to a radius. The cutoff selects the **model**, so changing it creates a different parameter space. CS has no knowledge of the eventual supercell size.

## Supercell: the explicit atomic structure

```python
from mlfcs import Supercell

supercell_atoms = primitive_atoms.repeat((3, 3, 3))
supercell = Supercell.from_atoms(space.primitive, supercell_atoms)

print(supercell.matrix)
print(len(supercell.numbers))
```

MLFCS does not choose or enlarge the supercell. You supply its ASE `Atoms` object, including its atom order. `Supercell.from_atoms` finds the integer basis relation $C_{\mathrm{s}} = M C_{\mathrm{p}}$ in ASE's row-vector convention; `supercell.matrix` is $M$. It checks the lattice and maps every supercell atom to exactly one primitive site and periodic translation, using `space.primitive.symprec` as a Cartesian length in Å. You do not supply $M$ separately.

Equivalent atom orderings can be mapped, but **training structures must preserve the order of the particular supplied supercell**. The supercell's periodic quotient labels identify primitive translations that refer to the same supercell atom.

## ClusterMap: the bridge and its rank check

```python
from mlfcs import ClusterMap

mapping = ClusterMap.build(space, supercell)
for order in space.orders:
    info = mapping.rank_info(order)
    print(order, info.rank, info.parameters, info.nullity)
    info.require_full()
```

The map folds each primitive orbit image onto supercell atom indices. It does not move the orbit definitions into the supercell. A small or unfortunate supercell can fold different primitive interactions onto the same observations. `rank_info(order)` checks this **structural identifiability** before expensive force calculations; `require_full()` raises `AliasingError` when parameters cannot be distinguished. Full structural rank does not guarantee that a particular fitting dataset has enough varied displacements.

One CS can be paired with different supercells by building a new `Supercell` and `ClusterMap` for each. After mapping, continue with [finite differences](finite-difference-api.md) or [force fitting](fitting-api.md).
