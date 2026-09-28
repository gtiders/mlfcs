# Core concepts

MLFCS uses ASE structures as the user-facing description of crystals and force data. The central model object is `ClusterSpace` (often called CS): it records the primitive motif and symmetry information needed downstream, then defines the force-constant clusters and parameters. An explicit `Supercell` supplies the structure on which forces are evaluated. `ClusterMap` connects the model to that supercell.

## `ClusterSpace`: define the primitive model

`ClusterSpace` is the normal starting point. Construct it directly from primitive ASE `Atoms`; it creates and retains the primitive-cell representation used internally for symmetry, lattice coordinates, masses, and downstream mapping.

```python
from ase.build import bulk
from mlfcs import ClusterSpace

si = bulk("Si", "diamond", a=5.43)
space = ClusterSpace(
    si,
    cutoffs={2: -7, 3: -6},
    max_body_orders={2: 2, 3: 3},
    symprec=1e-5,
)
print(space.orders, space.n_parameters)
```

The public constructor is:

```python
ClusterSpace(atoms, *, cutoffs, max_body_orders, symprec=1e-5)
```

| Parameter | Meaning |
| --- | --- |
| `atoms` | ASE `Atoms` describing the primitive motif, in the desired site order. The masses on this object are carried into the model. |
| `cutoffs` | Mapping from force-constant order to interaction cutoff. Positive values are radii in Å; negative integer values select a neighbor shell, such as `-7` for the seventh shell. Supply an entry for every configured order. |
| `max_body_orders` | Mapping from force-constant order to the maximum number of distinct sites in a cluster. Supply an entry for every configured order. |
| `symprec` | Symmetry and structure-mapping precision in Å, default `1e-5`. |

`ClusterSpace` exposes `primitive`, `symmetry`, `blocks`, `orbits`, `orders`, `pbc`, `n_parameters`, `parameter_offsets`, and `fingerprint`; `block(order)` returns the settings and parameter/orbit slices for an order. The internal `space.primitive` is a read-only representation of the useful primitive data that later algorithms need. It is not a second user construction step. The masses are inherited from the input ASE structure; to choose isotope or site masses before creating the model, set them on the ASE `Atoms` first.

Positive cutoffs are radii in Å. A negative integer selects a neighbor shell, for example `-7` for the seventh shell. Each order has its own cutoff and body-order limit, so harmonic and anharmonic blocks can be defined together.

## `Supercell`: provide the force-evaluation structure

`Supercell` describes the explicit ASE structure on which the model will be evaluated. Users provide the atoms and their order; MLFCS infers the integer basis relation and the primitive site and translation represented by each supercell atom.

```python
from mlfcs import Supercell

supercell_atoms = si.repeat((3, 3, 3))
supercell = Supercell.from_atoms(space.primitive, supercell_atoms)
print(supercell.matrix, supercell.determinant, len(supercell.numbers))
```

The constructor is `Supercell.from_atoms(primitive, atoms)`: `primitive` is `space.primitive`, and `atoms` is the explicit ASE supercell. There is no separate matrix argument. With ASE row-vector cells, the inferred relation is $C_s = M C_p$, and `supercell.matrix` is $M$.

The object exposes `primitive`, `matrix`, `cell`, `scaled_positions`, `numbers`, `sites`, `translations`, `quotients`, and `determinant`. These arrays follow the supplied supercell atom order. Preserve that order in displacement structures and force data. The primitive's `symprec` governs lattice and atomic mapping.

## `ClusterMap`: connect model and data geometry

`ClusterMap` folds each primitive cluster image onto the atom indices of one explicit supercell. Build one map for the chosen pair of `ClusterSpace` and `Supercell`:

```python
from mlfcs import ClusterMap

mapping = ClusterMap.build(space, supercell)
for order in space.orders:
    info = mapping.rank_info(order)
    print(order, info.rank, info.parameters, info.nullity, info.aliases)
    info.require_full()
```

The builder is `ClusterMap.build(space, supercell)`. The mapping exposes `space`, `supercell`, `atoms`, and `fingerprint`; `atoms` stores the supercell atom indices for each orbit's cluster images. Its `aliases(order)` method reports primitive images folded onto the same ordered atom tuple.

`rank_info(order=None)` returns a `RankInfo` with `parameters`, `rank`, `aliases`, `nullity`, and `full`. Omitting `order` combines the configured orders. `require_full()` raises `AliasingError` if the supercell cannot distinguish all parameters in the selected model block. This structural check is useful before force calculations. It is distinct from the question of whether a particular training set contains enough varied displacements.

The complete object flow is `ASE primitive Atoms → ClusterSpace`, `ASE supercell Atoms → Supercell`, then `(ClusterSpace, Supercell) → ClusterMap`. The map is shared by [finite differences](finite-difference.md) and [fitting](fitting.md). Anharmonic post-processing can continue with [SCPH](scph.md); stochastic finite-temperature force matching is described in [SSCHA](sscha.md).

## Si examples

The [Si finite-difference script](../tutorial/SI/finite-difference-fc2/run.py) shows `ClusterSpace`, explicit `SPOSCAR`, and `ClusterMap` in a compact FC2 workflow. The [Si fitting script](../tutorial/SI/fitting/fit.py) configures multiple orders and uses the same model-to-supercell mapping pattern.
