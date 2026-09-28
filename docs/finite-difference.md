# Finite differences

`FiniteDifference` defines one force-constant reconstruction experiment. It generates an ordered sequence of ASE structures; the force on each structure must remain associated with that exact geometry and atom order until reconstruction.

## Object and constructor

```python
FiniteDifference(mapping, *, order, disps=0.01)
```

| Parameter | Meaning |
| --- | --- |
| `mapping` | `ClusterMap` connecting the primitive model to the explicit supercell. |
| `order` | Force-constant order to reconstruct; it must be present in the cluster space and structurally identifiable in this supercell. One `FiniteDifference` handles one order. |
| `disps` | One positive displacement length in Å, or a sequence of distinct positive lengths. Default `0.01` Å. Multiple lengths invoke zero-displacement extrapolation. |

The object exposes `mapping`, `order`, `disps`, and `n_configurations`. `displacements()` yields the canonical structures in reconstruction order. If $N_k$ is the number of selected cluster keys, $N_d$ the number of displacement lengths, and $n$ the order, then `n_configurations = N_k N_d 2^{n-1}`.

## Minimal Si example with an ASE calculator

```python
from ase.calculators.emt import EMT
from ase.io import read
from mlfcs import ClusterMap, ClusterSpace, FiniteDifference, Supercell

primitive_atoms = read("POSCAR")
space = ClusterSpace(primitive_atoms, cutoffs={2: 6.0}, max_body_orders={2: 2})
supercell = Supercell.from_atoms(space.primitive, read("SPOSCAR"))
mapping = ClusterMap.build(space, supercell)

fd = FiniteDifference(mapping, order=2, disps=0.01)
evaluated = fd.evaluate(EMT())
fc2 = fd.reconstruct(evaluated)
```

`evaluate(calculator)` forces a fresh force calculation for every displacement and returns ASE structures with their forces stored. Any ASE calculator can provide the forces, including calculators wrapping machine-learning potentials such as MACE or NEP. The calculator must support the elements and structures in the chosen supercell.

## External DFT and other external force programs

When forces are produced outside ASE, use the same generated geometry sequence:

1. Iterate over `fd.displacements()` and write each structure in a format that preserves its cell and atom order.
2. Run the external calculation for each structure.
3. Read the resulting structures and forces back in the same order.
4. Attach each force array to its matching ASE structure, then reconstruct.

```python
from ase.calculators.singlepoint import SinglePointCalculator

for atoms, forces in zip(displaced_atoms, force_arrays, strict=True):
    atoms.calc = SinglePointCalculator(atoms, forces=forces)

fc2 = fd.reconstruct(displaced_atoms)
```

`force_arrays` is one `(n_atoms, 3)` force array per structure, in eV/Å. The reconstruction input is the ordered ASE structure sequence, not a bare force-array collection. The repository's [Si VASP example](../tutorial/SI/finite-difference-fc2/run.py) demonstrates writing ordered VASP structures, reading `vasprun.xml`, and reconstructing FC2.

The same pattern applies to an external machine-learning or electronic-structure program that does not expose an ASE calculator: write the structures, calculate forces, and return those forces on the corresponding ASE `Atoms` objects. If the potential already has an ASE calculator, `evaluate()` is the direct route.

## Multiple displacement lengths and extrapolation

`disps` may contain several lengths, for example:

```python
fd = FiniteDifference(mapping, order=3, disps=(0.005, 0.01, 0.015))
```

For each displacement pattern, reconstruction combines the central-difference estimates across the requested lengths and extrapolates to zero displacement using an even-error polynomial in displacement squared. This reduces finite-step truncation error when the force calculations are consistent across amplitudes. It also increases the number of structures in proportion to the number of lengths. Each length uses the same mapping, atom order, force units, and force source definition.

Extrapolation is independent of whether forces come from an ASE DFT calculator, an ASE machine-learning calculator, or an external calculation workflow. The [Si FC2/FC3 extrapolation example](../tutorial/SI/fc23-extrapolated/run.py) shows the multi-length workflow and its generated model files.

## Reconstruction and result

`fd.reconstruct(structures)` returns a `ForceConstants` containing the selected order. It verifies sequence length, atomic numbers and order, periodic cell and positions, and the presence and finiteness of stored forces. It does not launch calculations or infer a reordering.

```python
projection = fc2.enforce_asr(orders=(2,))
fc2_asr = projection.force_constants
fc2_asr.save("fc2-si.mlfcs")
fc2_asr.write("fc2-phonopy.hdf5", mapping, format="phonopy_hdf5", order=2)
```

ASR is an explicit subsequent projection. `ForceConstants` supports native save/load and format-specific export; see [force constants](force-constants.md).
