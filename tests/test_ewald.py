"""Physical contracts for the fixed-supercell dipole Ewald correction."""

from __future__ import annotations

from itertools import product
from math import pi

import h5py
import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from scipy.special import erfc

from mlfcs.cluster_space import ClusterSpace
from mlfcs.core import PrimitiveCell
from mlfcs.force_constants import write_phonopy
from mlfcs.force_constants.export import translated_atoms
from mlfcs.supercell import ClusterMap, Supercell
from mlfcs.tools import Ewald


def _mapping() -> tuple[ClusterSpace, ClusterMap]:
    atoms = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.diag([4.0, 4.0, 4.0]),
        pbc=True,
    )
    primitive = PrimitiveCell.from_atoms(atoms)
    space = ClusterSpace(atoms, cutoffs={2: 4.0}, max_body_orders={2: 2})
    supercell = Supercell.from_atoms(primitive, atoms.repeat((2, 1, 1)))
    return space, ClusterMap.build(space, supercell)


def _ewald(accuracy: float = 1e-10, *, eta: float | None = None) -> Ewald:
    space, mapping = _mapping()
    born = np.stack((np.eye(3), -np.eye(3)))
    return Ewald(space, mapping, born, np.diag([2.0, 3.0, 4.0]), accuracy=accuracy, eta=eta)


def _full(ewald: Ewald) -> np.ndarray:
    mapping = ewald.mapping
    folded = ewald.fc2
    return np.asarray(
        [
            folded[int(mapping.supercell.sites[first]), translated_atoms(mapping, first)]
            for first in range(len(mapping.supercell.numbers))
        ]
    )


def _atoms(ewald: Ewald) -> Atoms:
    supercell = ewald.mapping.supercell
    return Atoms(
        numbers=supercell.numbers,
        scaled_positions=supercell.scaled_positions,
        cell=supercell.cell,
        pbc=True,
    )


def test_fc2_is_finite_reciprocal_and_acoustic() -> None:
    ewald = _ewald()
    fc2 = ewald.fc2
    assert fc2.shape == (2, 4, 3, 3)
    assert np.all(np.isfinite(fc2))
    assert not np.any(np.isnan(fc2))
    full = _full(ewald)
    np.testing.assert_allclose(full, full.transpose(1, 0, 3, 2), rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.sum(full, axis=1), 0.0, rtol=0, atol=1e-10)
    assert np.max(np.abs(full)) > 0.0


def test_ase_energy_and_force_are_the_same_quadratic_form() -> None:
    ewald = _ewald()
    atoms = _atoms(ewald)
    atoms.positions[0, 0] += 0.02
    atoms.positions[2, 1] -= 0.01
    displacement = atoms.positions - _atoms(ewald).positions
    full = _full(ewald)
    expected = -np.einsum("ijab,jb->ia", full, displacement)
    atoms.calc = SinglePointCalculator(atoms, forces=np.ones_like(expected))
    np.testing.assert_allclose(ewald.get_forces(atoms), expected, rtol=0, atol=1e-11)
    np.testing.assert_array_equal(atoms.get_forces(), np.ones_like(expected))
    np.testing.assert_allclose(
        ewald.get_potential_energy(atoms), -0.5 * np.sum(displacement * expected), atol=1e-12
    )
    step = 1e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[1, 2] += step
    minus.positions[1, 2] -= step
    derivative = (ewald.get_potential_energy(plus) - ewald.get_potential_energy(minus)) / (2 * step)
    np.testing.assert_allclose(-derivative, expected[1, 2], atol=1e-10)
    rounded = atoms.copy()
    rounded.cell[0, 0] += 0.5 * ewald.space.primitive.symprec
    np.testing.assert_allclose(ewald.get_forces(rounded), expected, atol=1e-11)
    rounded.cell[0, 0] += ewald.space.primitive.symprec
    with pytest.raises(ValueError, match="cell residual"):
        ewald.get_forces(rounded)


def test_explicit_sum_can_be_written_only_as_phonopy_fc2(tmp_path) -> None:
    ewald = _ewald()
    total = np.zeros_like(ewald.fc2) + ewald.fc2
    file = write_phonopy(tmp_path / "fc2.hdf5", total, ewald.mapping, format="phonopy_hdf5")
    with h5py.File(file, "r") as handle:
        np.testing.assert_allclose(handle["force_constants"][:], _full(ewald), atol=1e-8)


def test_accuracy_and_input_validation() -> None:
    coarse = _ewald(1e-7)
    fine = _ewald(1e-11)
    np.testing.assert_allclose(coarse.fc2, fine.fc2, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        fine.fc2,
        _ewald(1e-11, eta=0.5 * fine._eta).fc2,
        rtol=1e-8,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        fine.fc2,
        _ewald(1e-11, eta=1.5 * fine._eta).fc2,
        rtol=1e-8,
        atol=1e-9,
    )
    space, mapping = _mapping()
    with pytest.raises(ValueError, match="positive definite"):
        Ewald(space, mapping, np.zeros((2, 3, 3)), np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="born"):
        Ewald(space, mapping, np.zeros((3, 3, 3)), np.eye(3))
    with pytest.raises(ValueError, match="accuracy"):
        Ewald(space, mapping, np.zeros((2, 3, 3)), np.eye(3), accuracy=0.0)
    with pytest.raises(ValueError, match="eta"):
        Ewald(space, mapping, np.zeros((2, 3, 3)), np.eye(3), eta=0.0)


def test_dilute_pair_tends_to_the_analytic_dipole_tensor() -> None:
    atoms = Atoms(
        "NaCl",
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=np.diag([80.0, 80.0, 80.0]),
        pbc=True,
    )
    primitive = PrimitiveCell.from_atoms(atoms)
    space = ClusterSpace(atoms, cutoffs={2: 3.0}, max_body_orders={2: 2})
    mapping = ClusterMap.build(space, Supercell.from_atoms(primitive, atoms))
    ewald = Ewald(space, mapping, [np.eye(3), -np.eye(3)], np.eye(3))
    from ase import units

    coulomb = units.Hartree * units.Bohr
    expected = -coulomb * np.diag([-2.0, 1.0, 1.0]) / 2.0**3
    np.testing.assert_allclose(ewald.fc2[0, 1], expected, rtol=2e-4, atol=2e-4)


def test_equivalent_supercell_bases_have_the_same_correction() -> None:
    first = _ewald()
    space = first.space
    atoms = _atoms(first)
    atoms.set_cell(np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ atoms.cell)
    remapped = ClusterMap.build(space, Supercell.from_atoms(space.primitive, atoms))
    second = Ewald(space, remapped, first.born, first.dielectric)
    np.testing.assert_allclose(first.fc2, second.fc2, rtol=0, atol=1e-10)


def test_dipole_tensor_is_the_negative_hessian_of_the_periodic_green_function() -> None:
    ewald = _ewald()
    cell = ewald.mapping.supercell.cell
    r = _atoms(ewald).positions[1] - _atoms(ewald).positions[0]
    epsilon = ewald.dielectric
    inverse = np.linalg.inv(epsilon)
    determinant = np.linalg.det(epsilon)
    eta = ewald._eta
    indices = np.asarray(tuple(product(range(-8, 9), repeat=3)))
    real_vectors = indices @ cell
    reciprocal = indices @ (2 * pi * np.linalg.inv(cell).T)
    reciprocal_norm = np.einsum("ni,ij,nj->n", reciprocal, epsilon, reciprocal)
    nonzero = reciprocal_norm > 0
    reciprocal = reciprocal[nonzero]
    reciprocal_norm = reciprocal_norm[nonzero]

    def green(position):
        vectors = real_vectors + position
        distances = np.sqrt(np.einsum("ni,ij,nj->n", vectors, inverse, vectors))
        real = np.sum(erfc(eta * distances) / distances) / np.sqrt(determinant)
        wave = (4 * pi / abs(np.linalg.det(cell))) * np.sum(
            np.exp(-reciprocal_norm / (4 * eta**2))
            * np.cos(reciprocal @ position)
            / reciprocal_norm
        )
        return units.Hartree * units.Bohr * (real + wave)

    step = 1e-3
    hessian = np.empty((3, 3))
    for first in range(3):
        for second in range(3):
            left = np.zeros(3)
            right = np.zeros(3)
            left[first] = step
            right[second] = step
            hessian[first, second] = (
                green(r + left + right)
                - green(r + left - right)
                - green(r - left + right)
                + green(r - left - right)
            ) / (4 * step**2)
    # Z_Na = I and Z_Cl = -I, so the Born contraction changes -Hess(G) to Hess(G).
    np.testing.assert_allclose(ewald.fc2[0, 1], hessian, rtol=1e-5, atol=2e-6)
