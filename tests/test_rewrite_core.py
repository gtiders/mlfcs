"""Focused contracts for the isolated rewritten primitive-cell core."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase import Atoms
from ase.build import bulk

from mlfcs.core import LatticeSite, PrimitiveCell, PrimitiveFrame, PrimitiveSymmetry
from mlfcs.core.algebra.exact import certified_rank, saturated_kernel, verify_kernel
from mlfcs.core.algebra.integer import exact_product, rotate_q_labels
from mlfcs.core.geometry import PeriodicGeometry, unique_distances


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
    assert not np.allclose(primitive.cartesian_positions, 0.0)


def test_primitive_cell_refuses_nonperiodic_or_singular_input() -> None:
    nonperiodic = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=False)
    singular = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.zeros((3, 3)), pbc=True)

    with pytest.raises(ValueError, match="periodic"):
        PrimitiveCell.from_atoms(nonperiodic, symprec=1e-5)
    with pytest.raises(ValueError, match="nonsingular"):
        PrimitiveCell.from_atoms(singular, symprec=1e-5)

    repeated = bulk("Si", "diamond", a=5.43).repeat((2, 1, 1))
    with pytest.raises(ValueError, match="primitive cell contains"):
        PrimitiveCell.from_atoms(repeated, symprec=1e-5)


def test_primitive_frame_is_invariant_to_an_integer_shear() -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    change = np.asarray([[1, 7, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    inverse = np.asarray([[1, -7, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    sheared = Atoms(
        numbers=atoms.numbers,
        scaled_positions=np.mod(atoms.get_scaled_positions(wrap=False) @ inverse, 1.0),
        cell=change @ np.asarray(atoms.cell),
        pbc=True,
    )
    plain = PrimitiveFrame.from_primitive(PrimitiveCell.from_atoms(atoms, symprec=1e-5))
    transformed = PrimitiveFrame.from_primitive(PrimitiveCell.from_atoms(sheared, symprec=1e-5))
    translations = np.asarray([[11, -4, 3], [-8, 2, 5]], dtype=np.int64)

    np.testing.assert_allclose(plain.canonical.cell, transformed.canonical.cell, atol=1e-12)
    np.testing.assert_allclose(
        plain.canonical.scaled_positions,
        transformed.canonical.scaled_positions,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        transformed.to_source_translations(transformed.to_canonical_translations(translations)),
        translations,
    )


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


def test_periodic_geometry_uses_true_mic_and_one_declared_precision() -> None:
    cell = np.asarray([[1.0, 0.0, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    geometry = PeriodicGeometry(cell)
    vector = np.asarray([0.76, 0.49, 0.0])
    image, length = geometry.minimum_image(vector)
    shifts = np.stack(
        np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), np.arange(-2, 3), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    brute = vector + shifts @ cell

    assert length == pytest.approx(float(np.min(np.linalg.norm(brute, axis=1))))
    assert np.linalg.norm(image) == pytest.approx(length)
    assert len(PeriodicGeometry(np.eye(3)).closest_images([0.5, 0.0, 0.0], symprec=1e-8)[0]) == 2
    assert unique_distances([0.0, 1.0, 1.0 + 0.5e-5, 2.0], symprec=1e-5) == (1.0, 2.0)
