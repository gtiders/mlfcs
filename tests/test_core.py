"""Focused contracts for the isolated rewritten primitive-cell core."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase import Atoms
from ase.build import bulk
from ase.data import atomic_masses
from ase.geometry import find_mic

from mlfcs.core import LatticeSite, PrimitiveCell, PrimitiveSymmetry
from mlfcs.core.algebra.exact import certified_rank, saturated_kernel, verify_kernel
from mlfcs.core.algebra.integer import exact_product, rotate_q_labels
from mlfcs.core.geometry import PeriodicGeometry


def test_periodic_geometry_finds_skewed_images_and_all_nearby_shifts() -> None:
    cell = np.asarray([[1.0, 0.0, 0.0], [0.5, 3**0.5 / 2, 0.0], [0.0, 0.0, 5.0]])
    geometry = PeriodicGeometry(cell)
    vector = np.asarray([0.49, 0.49, 0.0]) @ cell
    wrapped = (np.asarray([0.49, 0.49, 0.0]) - np.rint([0.49, 0.49, 0.0])) @ cell
    minimum, length = geometry.minimum_image(vector)
    expected, expected_length = find_mic(vector, cell)

    assert np.linalg.norm(wrapped) > length
    np.testing.assert_allclose(minimum, expected, atol=1e-14)
    assert length == pytest.approx(expected_length)
    images, shifts = geometry.closest_images(vector, symprec=1e-5)
    assert {tuple(shift) for shift in shifts} == {(-1, 0, 0), (0, -1, 0)}
    np.testing.assert_allclose(np.linalg.norm(images, axis=1), length, atol=1e-14)

    values = np.stack([vector, -vector])
    images, lengths = geometry.minimum_image(values)
    _, oracle_lengths = find_mic(values, cell)
    np.testing.assert_allclose(lengths, oracle_lengths, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(images, axis=1), lengths, atol=1e-14)

    rng = np.random.default_rng(23)
    for shear in (0.0, 2.0, 11.0):
        skewed = np.asarray([[1.0, 0.0, 0.0], [shear, 1.2, 0.0], [0.3, 0.4, 1.4]])
        batch = rng.uniform(-4.0, 4.0, size=(32, 3)) @ skewed
        _, actual_lengths = PeriodicGeometry(skewed).minimum_image(batch)
        _, expected_lengths = find_mic(batch, skewed)
        np.testing.assert_allclose(actual_lengths, expected_lengths, atol=1e-13)

    unit = PeriodicGeometry(np.eye(3))
    _, nearby = unit.closest_images(np.zeros(3), symprec=2.1)
    assert (2, 0, 0) in {tuple(shift) for shift in nearby}


def test_rewritten_core_has_no_dependency_on_existing_mlfcs_layers() -> None:
    assert internal_dependencies("core") == set()


def test_primitive_cell_owns_a_normalized_read_only_snapshot() -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_scaled_positions(atoms.get_scaled_positions() + [2.0, -1.0, 0.0])
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    atoms.positions[:] = 0.0

    assert np.all((primitive.scaled_positions >= 0.0) & (primitive.scaled_positions < 1.0))
    assert not primitive.cell.flags.writeable
    assert not primitive.scaled_positions.flags.writeable
    assert not primitive.numbers.flags.writeable
    assert not primitive.masses.flags.writeable
    np.testing.assert_array_equal(primitive.masses, atomic_masses[primitive.numbers])
    assert not np.allclose(primitive.cartesian_positions, 0.0)


def test_primitive_masses_are_per_site_and_immutable() -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    masses = [28.0, 29.0]
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5, masses=masses)
    masses[0] = 99.0
    np.testing.assert_array_equal(primitive.masses, [28.0, 29.0])
    np.testing.assert_array_equal(primitive.to_atoms().get_masses(), [28.0, 29.0])
    assert primitive.fingerprint == primitive.with_masses([30.0, 31.0]).fingerprint
    with pytest.raises(ValueError, match="one value per atom"):
        primitive.with_masses([28.0])
    with pytest.raises(ValueError, match="positive finite"):
        primitive.with_masses([28.0, np.nan])


def test_primitive_cell_refuses_nonperiodic_or_singular_input() -> None:
    nonperiodic = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=False)
    singular = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.zeros((3, 3)), pbc=True)

    with pytest.raises(ValueError, match="periodic"):
        PrimitiveCell.from_atoms(nonperiodic, symprec=1e-5)
    nonperiodic.pbc = (True, True, False)
    with pytest.raises(ValueError, match="three-dimensional periodic"):
        PrimitiveCell.from_atoms(nonperiodic, symprec=1e-5)
    with pytest.raises(ValueError, match="nonsingular"):
        PrimitiveCell.from_atoms(singular, symprec=1e-5)

    repeated = bulk("Si", "diamond", a=5.43).repeat((2, 1, 1))
    with pytest.raises(ValueError, match="primitive cell contains"):
        PrimitiveCell.from_atoms(repeated, symprec=1e-5)


def test_primitive_periodicity_is_inherited_by_copies() -> None:
    primitive = PrimitiveCell.from_atoms(bulk("Ar", "sc", a=1.0))
    assert primitive.pbc == (True, True, True)
    assert tuple(primitive.to_atoms().pbc) == primitive.pbc
    assert primitive.with_masses([40.0]).pbc == primitive.pbc


def test_primitive_symmetry_is_a_closed_action_on_the_motif() -> None:
    primitive = PrimitiveCell.from_atoms(bulk("Si", "diamond", a=5.43), symprec=1e-5)
    symmetry = PrimitiveSymmetry.from_primitive(primitive)

    assert symmetry.size > 1
    np.testing.assert_array_equal(
        np.sort(symmetry.site_permutations, axis=1),
        np.broadcast_to(np.arange(primitive.size), symmetry.site_permutations.shape),
    )
    for operation in range(symmetry.size):
        label = LatticeSite(0, (2, -3, 1))
        mapped = symmetry.transform_site(operation, label)
        source = primitive.scaled_positions[label.site] + label.translation
        target = primitive.scaled_positions[mapped.site] + mapped.translation
        expected = source @ symmetry.rotations[operation].T + symmetry.translations[operation]
        np.testing.assert_allclose(target, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(("residual", "accepted"), ((0.5e-5, True), (5e-5, False)))
def test_symprec_is_the_only_geometric_mapping_threshold(
    monkeypatch: pytest.MonkeyPatch, residual: float, accepted: bool
) -> None:
    primitive = PrimitiveCell.from_atoms(bulk("Ar", "sc", a=1.0), symprec=1e-5)
    dataset = SimpleNamespace(
        rotations=np.eye(3, dtype=np.int32)[None, :, :],
        translations=np.array([[residual, 0.0, 0.0]]),
        international="P1",
    )
    monkeypatch.setattr(
        "mlfcs.core.symmetry.spglib.get_symmetry_dataset",
        lambda cell, symprec: dataset,
    )

    if accepted:
        assert PrimitiveSymmetry.from_primitive(primitive).size == 1
    else:
        with pytest.raises(ValueError, match="nearest residual"):
            PrimitiveSymmetry.from_primitive(primitive)


def test_symmetry_mapping_uses_the_cartesian_nearest_image_in_a_sheared_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = np.asarray([[1.0, 0.0, 0.0], [10.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    primitive = PrimitiveCell.from_atoms(
        Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True), symprec=0.1
    )
    translation = np.asarray([0.0, 0.055, 0.0]) @ np.linalg.inv(cell)
    dataset = SimpleNamespace(
        rotations=np.eye(3, dtype=np.int32)[None, :, :],
        translations=translation[None, :],
        international="P1",
    )
    monkeypatch.setattr(
        "mlfcs.core.symmetry.spglib.get_symmetry_dataset", lambda *args, **kwargs: dataset
    )

    symmetry = PrimitiveSymmetry.from_primitive(primitive)
    np.testing.assert_array_equal(symmetry.site_shifts[0, 0], [0, 0, 0])


def test_exact_algebra_handles_large_integers_and_certifies_a_kernel() -> None:
    scale = 2**40
    product = exact_product(
        np.asarray([[scale, scale]], dtype=np.int64),
        np.asarray([[scale], [scale]], dtype=np.int64),
    )
    assert int(product[0, 0]) == 2 * scale**2

    matrix = np.asarray([[1, -2, 1], [2, 1, 1]], dtype=np.int64)
    kernel = saturated_kernel(matrix)
    assert certified_rank(matrix) == 2
    assert kernel.dtype == object
    verify_kernel(matrix, kernel)
    assert kernel.shape[1] == matrix.shape[1] - certified_rank(matrix)

    large_kernel = saturated_kernel([[2**70, 1]])
    verify_kernel([[2**70, 1]], large_kernel)
    assert any(abs(value) > 2**63 for value in large_kernel.flat)
    assert all(type(value) is int for value in large_kernel.flat)
    with pytest.raises(ValueError, match="not an integer"):
        certified_rank(np.asarray([[1.0]]))


def test_reciprocal_labels_use_the_exact_dual_of_the_primitive_rotation() -> None:
    rotation = np.asarray([[1, 7, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    labels = np.asarray([[2, -1, 4], [-3, 5, 0]], dtype=np.int64)
    moved = rotate_q_labels(labels, rotation)

    np.testing.assert_array_equal(moved @ rotation, labels)
    np.testing.assert_array_equal(rotate_q_labels(labels, rotation, denominator=11), moved % 11)
    second = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    np.testing.assert_array_equal(
        rotate_q_labels(rotate_q_labels(labels, rotation), second),
        rotate_q_labels(labels, second @ rotation),
    )
    with pytest.raises(TypeError, match="integers"):
        rotate_q_labels(labels.astype(float), rotation)
    with pytest.raises(ValueError, match="positive integer"):
        rotate_q_labels(labels, rotation, denominator=0)


def test_reciprocal_label_rotation_does_not_wrap_int64_products() -> None:
    scale = 2**40
    rotation = np.asarray([[1, scale, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    moved = rotate_q_labels(np.asarray([scale, 0, 0], dtype=np.int64), rotation)
    assert tuple(int(value) for value in moved) == (scale, -(scale**2), 0)
    np.testing.assert_array_equal(
        rotate_q_labels([scale, 0, 0], rotation, denominator=7),
        np.asarray([scale % 7, -(scale**2) % 7, 0]),
    )
