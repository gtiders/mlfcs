"""Small contracts for ASE-only finite differences."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from mlfcs.cluster_space import build_cluster_space
from mlfcs.core import PrimitiveCell
from mlfcs.core.errors import AliasingError
from mlfcs.finite_difference import FiniteDifference
from mlfcs.supercell import ClusterMap, Supercell


class HarmonicCalculator(Calculator):
    """An arbitrary ASE calculator with unit Cartesian on-site FC2."""

    implemented_properties: ClassVar = ["forces"]

    def __init__(self, positions: np.ndarray):
        super().__init__()
        self.positions = np.asarray(positions, dtype=float)
        self.calls = 0

    def calculate(
        self,
        atoms=None,
        properties=("forces",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results["forces"] = -(atoms.positions - self.positions)


def ar_mapping(repetitions: int, cutoff: float = 1.1) -> tuple[ClusterMap, Atoms]:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    space = build_cluster_space(
        primitive,
        cutoffs={2: cutoff},
        max_body_orders={2: 2},
    )
    supercell_atoms = atoms.repeat((repetitions,) * 3)
    cell = Supercell.from_atoms(
        primitive,
        supercell_atoms,
        matrix=np.diag([repetitions] * 3),
    )
    return ClusterMap.build(space, cell), supercell_atoms


def representative_tensors(model, order: int) -> list[np.ndarray]:
    block = model.space.block(order)
    values = model.coefficients[order]
    tensors = []
    offset = 0
    for orbit in model.space.orbits[block.orbits]:
        stop = offset + orbit.dimension
        tensors.append((orbit.component_basis @ values[offset:stop]).reshape((3,) * order))
        offset = stop
    assert offset == len(values)
    return tensors


def stored(structures, forces) -> tuple[Atoms, ...]:
    result = []
    for atoms, values in zip(structures, forces, strict=True):
        atoms.calc = SinglePointCalculator(atoms, forces=np.asarray(values))
        result.append(atoms)
    return tuple(result)


def test_evaluate_forces_every_calculation_and_freezes_standard_ase_forces(tmp_path) -> None:
    mapping, supercell_atoms = ar_mapping(3)
    fd = FiniteDifference(mapping, order=2, disps=(0.01,))
    calculator = HarmonicCalculator(supercell_atoms.positions)

    evaluated = fd.evaluate(calculator)

    assert calculator.calls == fd.n_configurations
    assert all(isinstance(atoms.calc, SinglePointCalculator) for atoms in evaluated)
    write(tmp_path / "forces.extxyz", evaluated)
    restored = tuple(read(tmp_path / "forces.extxyz", ":"))
    first = fd.reconstruct(evaluated)
    second = fd.reconstruct(restored)
    np.testing.assert_array_equal(second.coefficients[2], first.coefficients[2])

    block = first.space.block(2)
    tensors = representative_tensors(first, 2)
    onsite = [
        len(set(orbit.representative.sites)) == 1 for orbit in first.space.orbits[block.orbits]
    ]
    assert onsite.count(True) == 1
    np.testing.assert_allclose(tensors[onsite.index(True)], np.eye(3), atol=1e-14, rtol=0.0)
    for tensor, is_onsite in zip(tensors, onsite, strict=True):
        if not is_onsite:
            np.testing.assert_allclose(tensor, 0.0, atol=1e-14, rtol=0.0)


def test_finite_difference_has_no_persistence_contract() -> None:
    import mlfcs.finite_difference as module

    mapping, _ = ar_mapping(3)
    fd = FiniteDifference(mapping, order=2)

    assert fd.disps == (0.01,)
    assert not hasattr(module, "load_fd")
    assert not hasattr(fd, "save")
    assert not hasattr(fd, "file")
    assert not hasattr(fd, "fingerprint")
    with pytest.raises(TypeError, match="steps"):
        FiniteDifference(mapping, order=2, steps=(0.01,))


def test_reconstruct_accepts_only_ordered_atoms_with_stored_forces() -> None:
    mapping, supercell_atoms = ar_mapping(3)
    fd = FiniteDifference(mapping, order=2)
    structures = tuple(fd.displacements())
    forces = [-(atoms.positions - supercell_atoms.positions) for atoms in structures]
    evaluated = stored(structures, forces)

    model = fd.reconstruct(evaluated)
    assert model.orders == (2,)
    assert model.space.fingerprint == mapping.space.fingerprint

    with pytest.raises(TypeError, match="ASE Atoms"):
        fd.reconstruct(np.asarray(forces))
    with pytest.raises(ValueError, match="expected"):
        fd.reconstruct(evaluated[:-1])
    missing = list(evaluated)
    missing[0] = missing[0].copy()
    with pytest.raises(ValueError, match="no stored ASE forces"):
        fd.reconstruct(missing)
    shuffled = list(evaluated)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    with pytest.raises(ValueError, match="does not match"):
        fd.reconstruct(shuffled)


def test_multiple_disps_remove_the_leading_even_difference_error() -> None:
    mapping, supercell_atoms = ar_mapping(1, cutoff=0.1)
    stiffness = 2.5
    cubic_error = 7.0
    fd = FiniteDifference(mapping, order=2, disps=(0.02, 0.01))
    structures = tuple(fd.displacements())
    forces = []
    for atoms in structures:
        displacement = atoms.positions - supercell_atoms.positions
        forces.append(-stiffness * displacement - cubic_error * displacement**3)

    model = fd.reconstruct(stored(structures, forces))
    tensors = representative_tensors(model, 2)
    assert len(tensors) == 1
    np.testing.assert_allclose(tensors[0], stiffness * np.eye(3), atol=1e-12, rtol=0.0)


def test_fd_rejects_an_aliased_supercell_without_writing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    mapping, _ = ar_mapping(1)

    with pytest.raises(AliasingError, match="nullity"):
        FiniteDifference(mapping, order=2)

    assert list(tmp_path.iterdir()) == []


def test_fc3_reconstruction_uses_atoms_and_the_orbit_basis() -> None:
    atoms = Atoms(
        numbers=[1, 2],
        scaled_positions=[[0.0, 0.0, 0.0], [0.17, 0.31, 0.23]],
        cell=[[3.1, 0.2, 0.1], [0.1, 3.7, 0.3], [0.2, 0.1, 4.2]],
        pbc=True,
    )
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    space = build_cluster_space(
        primitive,
        cutoffs={3: 0.01},
        max_body_orders={3: 1},
    )
    cell = Supercell.from_atoms(primitive, atoms, matrix=np.eye(3, dtype=np.int64))
    fd = FiniteDifference(ClusterMap.build(space, cell), order=3, disps=(0.001,))
    diagonal = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    structures = tuple(fd.displacements())
    forces = []
    for displaced in structures:
        movement = displaced.positions - atoms.positions
        forces.append(-0.5 * diagonal * movement**2)

    model = fd.reconstruct(stored(structures, forces))
    tensors = representative_tensors(model, 3)
    expected = np.zeros((2, 3, 3, 3))
    for atom in range(2):
        for axis in range(3):
            expected[atom, axis, axis, axis] = diagonal[atom, axis]
    np.testing.assert_allclose(tensors, expected, atol=1e-12, rtol=0.0)
