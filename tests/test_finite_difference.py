"""Numerical behavior of finite difference."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from mlfcs.cluster_space import ClusterSpace
from mlfcs.dataset import ForceDataset
from mlfcs.finite_difference import FiniteDifference
from mlfcs.mapping import ClusterMap


class HarmonicCalculator(Calculator):
    """An arbitrary ASE calculator with unit Cartesian on-site FC2."""

    implemented_properties: ClassVar = ["forces"]

    def __init__(self, positions: np.ndarray):
        """Capture the reference (n_atoms, 3) Cartesian positions in angstrom."""
        super().__init__()
        self.positions = np.asarray(positions, dtype=float)
        self.calls = 0

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes) -> None:
        """Compute on-site forces F = -(r-r_reference) in eV/angstrom.

        The unit stiffness is 1 eV/angstrom**2. Store an (n_atoms, 3) force array
        in ASE results and increment the calculation counter."""
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results["forces"] = -(atoms.positions - self.positions)


def ar_mapping(repetitions: int, cutoff: float = 1.1) -> tuple[ClusterMap, Atoms]:
    """Construct a cubic Ar FC2 model and its repeated reference supercell.

    repetitions is the number of primitive cells along each axis, and cutoff
    is in angstrom. Return (ClusterMap, ASE Atoms) in the same atom order."""
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = atoms
    space = ClusterSpace(primitive, cutoffs={2: cutoff}, max_body_orders={2: 2}, symprec=1e-05)
    supercell_atoms = atoms.repeat((repetitions,) * 3)
    return (
        ClusterMap(space, supercell_atoms, supercell_matrix=np.diag([repetitions] * 3)),
        supercell_atoms,
    )


def representative_tensors(model, order: int) -> list[np.ndarray]:
    """Expand the physical parameters into representative Cartesian tensors.

    model is a ForceConstants object containing the requested order p. Return
    one ndarray of shape (3,)*p per orbit in block order, in eV/angstrom**p."""
    block = model.cluster_space.block(order)
    values = model.coefficients[order]
    tensors = []
    offset = 0
    for orbit in model.cluster_space.orbits[block.orbits]:
        stop = offset + orbit.dimension
        tensors.append((orbit.component_basis @ values[offset:stop]).reshape((3,) * order))
        offset = stop
    assert offset == len(values)
    return tensors


def stored(structures, forces) -> tuple[Atoms, ...]:
    """Attach supplied force arrays to displaced ASE structures.

    forces contains one (n_atoms, 3) array in eV/angstrom per structure. Return
    the structures as a tuple with SinglePointCalculator results attached."""
    result = []
    for atoms, values in zip(structures, forces, strict=True):
        atoms.calc = SinglePointCalculator(atoms, forces=np.asarray(values))
        result.append(atoms)
    return tuple(result)


def test_external_calculation_and_extxyz_reconstruct_the_same_fc2(tmp_path) -> None:
    """Recover the same Hessian from in-memory and external stored-force samples."""
    mapping, supercell_atoms = ar_mapping(3)
    fd = FiniteDifference(mapping, order=2, disps=(0.01,))
    calculator = HarmonicCalculator(supercell_atoms.positions)
    evaluated = []
    for atoms in fd.sow():
        calculator.calculate(atoms, properties=["forces"], system_changes=all_changes)
        atoms.calc = SinglePointCalculator(atoms, forces=calculator.results["forces"])
        evaluated.append(atoms)
    assert calculator.calls == fd.n_configurations
    assert all(isinstance(atoms.calc, SinglePointCalculator) for atoms in evaluated)
    write(tmp_path / "forces.extxyz", evaluated)
    restored = tuple(read(tmp_path / "forces.extxyz", ":"))
    first = fd.reap(ForceDataset(mapping, evaluated))
    second = fd.reap(ForceDataset(mapping, restored))
    np.testing.assert_array_equal(second.coefficients[2], first.coefficients[2])
    block = first.cluster_space.block(2)
    tensors = representative_tensors(first, 2)
    onsite = [
        len(set(orbit.representative.sites)) == 1
        for orbit in first.cluster_space.orbits[block.orbits]
    ]
    assert onsite.count(True) == 1
    np.testing.assert_allclose(tensors[onsite.index(True)], np.eye(3), atol=1e-14, rtol=0.0)
    for tensor, is_onsite in zip(tensors, onsite, strict=True):
        if not is_onsite:
            np.testing.assert_allclose(tensor, 0.0, atol=1e-14, rtol=0.0)


def test_finite_difference_reconstructs_stored_forces() -> None:
    """Verify finite difference reconstructs stored forces."""
    mapping, supercell_atoms = ar_mapping(3)
    fd = FiniteDifference(mapping, order=2)
    structures = tuple(fd.sow())
    forces = [-(atoms.positions - supercell_atoms.positions) for atoms in structures]
    evaluated = stored(structures, forces)
    model = fd.reap(ForceDataset(mapping, evaluated))
    assert model.orders == (2,)


def test_multiple_disps_remove_the_leading_even_difference_error() -> None:
    """Verify multiple disps remove the leading even difference error."""
    mapping, supercell_atoms = ar_mapping(1, cutoff=0.1)
    stiffness = 2.5
    cubic_error = 7.0
    fd = FiniteDifference(mapping, order=2, disps=(0.02, 0.01))
    structures = tuple(fd.sow())
    forces = []
    for atoms in structures:
        displacement = atoms.positions - supercell_atoms.positions
        forces.append(-stiffness * displacement - cubic_error * displacement**3)
    model = fd.reap(ForceDataset(mapping, stored(structures, forces)))
    tensors = representative_tensors(model, 2)
    assert len(tensors) == 1
    np.testing.assert_allclose(tensors[0], stiffness * np.eye(3), atol=1e-12, rtol=0.0)


def test_fc3_reconstruction_uses_atoms_and_the_orbit_basis() -> None:
    """Verify fc3 reconstruction uses atoms and the orbit basis."""
    atoms = Atoms(
        numbers=[1, 2],
        scaled_positions=[[0.0, 0.0, 0.0], [0.17, 0.31, 0.23]],
        cell=[[3.1, 0.2, 0.1], [0.1, 3.7, 0.3], [0.2, 0.1, 4.2]],
        pbc=True,
    )
    primitive = atoms
    space = ClusterSpace(primitive, cutoffs={3: 0.01}, max_body_orders={3: 1}, symprec=1e-05)
    fd = FiniteDifference(
        ClusterMap(space, atoms, supercell_matrix=np.eye(3, dtype=np.int64)),
        order=3,
        disps=(0.001,),
    )
    diagonal = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    structures = tuple(fd.sow())
    forces = []
    for displaced in structures:
        movement = displaced.positions - atoms.positions
        forces.append(-0.5 * diagonal * movement**2)
    model = fd.reap(ForceDataset(fd.cluster_map, stored(structures, forces)))
    tensors = representative_tensors(model, 3)
    expected = np.zeros((2, 3, 3, 3))
    for atom in range(2):
        for axis in range(3):
            expected[atom, axis, axis, axis] = diagonal[atom, axis]
    np.testing.assert_allclose(tensors, expected, atol=1e-12, rtol=0.0)


def test_fc4_reconstruction_consumes_the_shared_force_dataset():
    """Recover quartic energy derivatives from ordered cubic force samples."""
    atoms = bulk("Ar", "sc", a=2.0)
    space = ClusterSpace(atoms, cutoffs={4: 0.1})
    mapping = ClusterMap(space, atoms)
    fd = FiniteDifference(mapping, order=4, disps=0.001)
    structures = tuple(fd.sow())
    stiffness = 2.5
    forces = [-stiffness / 6 * (sample.positions - atoms.positions) ** 3 for sample in structures]
    model = fd.reap(ForceDataset(mapping, stored(structures, forces)))
    expected = np.zeros((3, 3, 3, 3))
    for axis in range(3):
        expected[axis, axis, axis, axis] = stiffness
    np.testing.assert_allclose(representative_tensors(model, 4)[0], expected, atol=1e-12)
