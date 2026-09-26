"""Small exact contracts for the reduced reciprocal grid."""

from __future__ import annotations

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase import Atoms
from ase.build import bulk
from ase.data import atomic_masses
from scipy.constants import angstrom, atomic_mass, electron_volt

from mlfcs.cluster_space import ClusterSpace
from mlfcs.core import PrimitiveCell, PrimitiveSymmetry
from mlfcs.core.algebra.integer import rotate_q_labels
from mlfcs.force_constants import ForceConstants
from mlfcs.reciprocal import Harmonic, QGrid, QStars, StarPlan
from mlfcs.supercell import Supercell


def test_nondiagonal_grid_is_the_exact_dual_and_survives_a_supercell_rebasis() -> None:
    matrix = np.asarray([[2, 1, 0], [0, 3, 0], [0, 0, 2]], dtype=np.int64)
    rebasis = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    grid = QGrid.from_matrix(matrix.astype(object))

    assert internal_dependencies("reciprocal") == {
        "core",
        "fitting",
        "force_constants",
        "supercell",
    }
    assert grid.denominator == 12
    assert grid.size == 12
    assert len({tuple(label) for label in grid.labels}) == 12
    assert np.all((grid.labels >= 0) & (grid.labels < grid.denominator))
    for label in grid.labels:
        assert all(
            int(value) % grid.denominator == 0
            for value in label.astype(object) @ matrix.T.astype(object)
        )
    np.testing.assert_array_equal(QGrid.from_matrix(rebasis @ matrix).labels, grid.labels)
    np.testing.assert_array_equal(grid.points * grid.denominator, grid.labels)


def test_cubic_stars_are_a_partition_with_exact_member_routes() -> None:
    primitive = PrimitiveCell.from_atoms(bulk("Ar", "sc", a=1.0), symprec=1e-5)
    symmetry = PrimitiveSymmetry.from_primitive(primitive)
    matrix = np.diag([2, 2, 2])
    supercell = Supercell.from_atoms(primitive, primitive.to_atoms().repeat((2, 2, 2)))
    stars = QStars.from_symmetry(QGrid.from_matrix(supercell.matrix), symmetry)
    sheared = QStars.from_symmetry(
        QGrid.from_matrix(np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ matrix),
        symmetry,
    )

    assert tuple(stars.weights) == (1, 3, 3, 1)
    assert tuple(stars.representatives) == (0, 1, 3, 7)
    np.testing.assert_array_equal(sheared.grid.labels, stars.grid.labels)
    np.testing.assert_array_equal(sheared.star_of, stars.star_of)
    assert int(np.sum(stars.weights)) == stars.grid.size
    np.testing.assert_array_equal(
        np.bincount(stars.star_of, minlength=len(stars.weights)), stars.weights
    )
    for member, label in enumerate(stars.grid.labels):
        representative = stars.grid.labels[stars.representatives[stars.star_of[member]]]
        operation = int(stars.operations[member])
        image = rotate_q_labels(representative, symmetry.rotations[operation])
        sign = -1 if stars.antiunitary[member] else 1
        shift = np.asarray(stars.member_shift(member), dtype=object)
        np.testing.assert_array_equal(
            sign * image.astype(object), label.astype(object) + stars.grid.denominator * shift
        )
    values = np.arange(len(stars.representatives))
    np.testing.assert_array_equal(stars.expand(values), values[stars.star_of])
    assert int(np.sum(stars.expand(values))) == int(stars.weights @ values)
    with pytest.raises(ValueError, match="one leading entry"):
        stars.expand([1])


def test_time_reversal_pairs_only_the_labels_it_reaches() -> None:
    primitive = PrimitiveCell.from_atoms(bulk("Ar", "sc", a=1.0), symprec=1e-5)
    identity = PrimitiveSymmetry(
        rotations=np.eye(3, dtype=np.int32)[None, :, :],
        translations=np.zeros((1, 3)),
        cartesian_rotations=np.eye(3)[None, :, :],
        site_permutations=np.asarray([[0]], dtype=np.int32),
        site_shifts=np.zeros((1, 1, 3), dtype=np.int32),
        symbol="P1",
        symprec=primitive.symprec,
    )
    grid = QGrid.from_matrix(np.diag([3, 1, 1]))
    without = QStars.from_symmetry(grid, identity, time_reversal=False)
    with_reversal = QStars.from_symmetry(grid, identity, time_reversal=True)

    assert tuple(without.weights) == (1, 1, 1)
    assert tuple(with_reversal.weights) == (1, 2)
    assert tuple(with_reversal.antiunitary) == (False, False, True)
    assert with_reversal.member_shift(2) == (-1, 0, 0)


def test_only_grid_preserving_rotations_enter_a_nondiagonal_star() -> None:
    primitive = PrimitiveCell.from_atoms(bulk("Ar", "sc", a=1.0), symprec=1e-5)
    symmetry = PrimitiveSymmetry.from_primitive(primitive)
    grid = QGrid.from_matrix([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    stars = QStars.from_symmetry(grid, symmetry)
    lookup = {tuple(label): index for index, label in enumerate(grid.labels)}

    assert int(np.sum(stars.weights)) == grid.size
    assert len(stars.star_of) == grid.size
    assert any(
        tuple(
            int(value) for value in rotate_q_labels(label, rotation, denominator=grid.denominator)
        )
        not in lookup
        for rotation in symmetry.rotations
        for label in grid.labels
    )
    for member in range(grid.size):
        representative = int(stars.representatives[stars.star_of[member]])
        image = rotate_q_labels(
            grid.labels[representative],
            symmetry.rotations[stars.operations[member]],
            denominator=grid.denominator,
        )
        if stars.antiunitary[member]:
            image = -image % grid.denominator
        np.testing.assert_array_equal(image, grid.labels[member])


def test_harmonic_spectrum_uses_ase_mass_and_only_irreducible_qpoints() -> None:
    primitive = PrimitiveCell.from_atoms(bulk("Ar", "sc", a=1.0), symprec=1e-5)
    space = ClusterSpace(
        primitive.to_atoms(), symprec=primitive.symprec, cutoffs={2: 1.1}, max_body_orders={2: 2}
    )
    spring = 1.5
    coefficients = []
    for orbit in space.orbits:
        pair = orbit.representative.sites
        if pair[0] == pair[1]:
            tensor = 2 * spring * np.eye(3)
        else:
            direction = np.asarray(pair[1].translation, dtype=float)
            direction /= np.linalg.norm(direction)
            tensor = -spring * np.outer(direction, direction)
        parameters = np.linalg.lstsq(orbit.component_basis, tensor.reshape(-1), rcond=None)[0]
        np.testing.assert_allclose(orbit.component_basis @ parameters, tensor.reshape(-1))
        coefficients.extend(parameters)
    model = ForceConstants(space, {2: np.asarray(coefficients)})
    harmonic = Harmonic(model)
    stars = QStars.from_symmetry(QGrid.from_matrix(np.diag([4, 1, 1])), space.symmetry)

    frequencies = harmonic.frequencies(stars)
    assert frequencies.shape == (len(stars.representatives), 3)
    assert len(frequencies) < stars.grid.size
    factor = np.sqrt(electron_volt / (angstrom**2 * atomic_mass)) / (2 * np.pi * 1e12)
    expected = (
        np.sqrt(2 * spring * (1 - np.cos(2 * np.pi * stars.points[:, 0])) / atomic_masses[18])
        * factor
    )
    np.testing.assert_allclose(frequencies[:, 2], expected, atol=1e-6)
    np.testing.assert_allclose(frequencies[:, :2], 0.0, atol=1e-6)
    np.testing.assert_allclose(harmonic.frequencies([0.25, 0, 0])[-1], expected[1], atol=1e-6)
    heavier = primitive.with_masses([2 * primitive.masses[0]])
    heavier_space = ClusterSpace(
        heavier.to_atoms(), symprec=heavier.symprec, cutoffs={2: 1.1}, max_body_orders={2: 2}
    )
    heavier_model = ForceConstants(heavier_space, {2: model.coefficients[2]})
    np.testing.assert_allclose(
        Harmonic(heavier_model).frequencies(stars), frequencies / np.sqrt(2), atol=1e-6
    )
    matrices = harmonic.matrices(stars)
    assert matrices.shape == (len(stars.representatives), 3, 3)
    members = StarPlan.from_stars(stars, primitive).iter_matrices(matrices)
    for member, matrix in members:
        np.testing.assert_allclose(
            np.linalg.eigvalsh(matrix),
            np.linalg.eigvalsh(matrices[stars.star_of[member]]),
            atol=1e-12,
        )


def test_time_reversal_expansion_includes_the_reduced_label_gauge() -> None:
    atoms = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.25, 0, 0]],
        cell=np.diag([2.0, 3.0, 4.0]),
        pbc=True,
    )
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    symmetry = PrimitiveSymmetry.from_primitive(primitive)
    stars = QStars.from_symmetry(QGrid.from_matrix(np.diag([3, 1, 1])), symmetry)
    plan = StarPlan.from_stars(stars, primitive)
    member = next(index for index in range(stars.grid.size) if stars.antiunitary[index])
    assert stars.member_shift(member) == (-1, 0, 0)
    source = np.zeros((6, 6), dtype=complex)
    source[0, 3] = 1 + 2j
    source[3, 0] = 1 - 2j

    actual = plan.matrix(member, source)
    # q_image = -1/3, q_stored = 2/3.  The positional gauge at the second
    # site is exp(-2 pi i * 1/4) = -i, not one.
    assert actual[0, 3] == pytest.approx(2 + 1j)
    assert actual[3, 0] == pytest.approx(2 - 1j)
    assert not np.allclose(actual, source.conj())
    np.testing.assert_allclose(actual, actual.conj().T)
    np.testing.assert_allclose(np.linalg.eigvalsh(actual), np.linalg.eigvalsh(source))


def test_star_plan_rotates_and_streams_only_representative_matrices() -> None:
    primitive = PrimitiveCell.from_atoms(bulk("GaAs", "zincblende", a=5.65), symprec=1e-5)
    symmetry = PrimitiveSymmetry.from_primitive(primitive)
    stars = QStars.from_symmetry(QGrid.from_matrix(np.diag([3, 2, 2])), symmetry)
    plan = StarPlan.from_stars(stars, primitive)
    count = 3 * primitive.size
    source = np.arange(count * count, dtype=float).reshape(count, count)
    source = source + source.T + 1j * (source - source.T)
    matrices = np.repeat(source[None], len(stars.representatives), axis=0)
    expanded = list(plan.iter_matrices(matrices))

    assert len(expanded) == stars.grid.size
    assert [member for member, _ in expanded] == list(range(stars.grid.size))
    for member, matrix in expanded:
        np.testing.assert_allclose(matrix, matrix.conj().T)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(matrix), np.linalg.eigvalsh(source), atol=1e-12
        )
        operation = int(stars.operations[member])
        permutation = symmetry.site_permutations[operation]
        rotation = symmetry.cartesian_rotations[operation]
        first, second = 0, 1
        target_first, target_second = permutation[[first, second]]
        gauge = plan.phases[member, target_first] * plan.phases[member, target_second].conj()
        block = source[3 * first : 3 * first + 3, 3 * second : 3 * second + 3]
        if stars.antiunitary[member]:
            block = block.conj()
        expected = gauge * rotation.T @ block @ rotation
        np.testing.assert_allclose(
            matrix[
                3 * target_first : 3 * target_first + 3, 3 * target_second : 3 * target_second + 3
            ],
            expected,
        )
    with pytest.raises(ValueError, match="representative matrices must have shape"):
        list(plan.iter_matrices(matrices[:-1]))


def test_harmonic_rejects_a_site_symmetry_that_exchanges_unequal_masses() -> None:
    primitive = PrimitiveCell.from_atoms(
        bulk("Si", "diamond", a=5.43), symprec=1e-5, masses=[28.0, 29.0]
    )
    space = ClusterSpace(
        primitive.to_atoms(), symprec=primitive.symprec, cutoffs={2: 3.0}, max_body_orders={2: 2}
    )
    coefficients = np.zeros(space.block(2).parameters.stop - space.block(2).parameters.start)
    with pytest.raises(ValueError, match="different masses"):
        Harmonic(ForceConstants(space, {2: coefficients}))
