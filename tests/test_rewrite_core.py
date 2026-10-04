"""Focused contracts for the isolated rewritten primitive-cell core."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase import Atoms
from ase.build import bulk

from mlfcs import ClusterSpace
from mlfcs.algebra.exact import exact_kernel, exact_rank, verify_kernel
from mlfcs.algebra.integer import exact_product, rotate_q_labels
from mlfcs.core import LatticeSite
from mlfcs.core.symmetry import discover_symmetry


def test_rewritten_core_has_no_dependency_on_existing_mlfcs_layers() -> None:
    assert internal_dependencies("core") == {"_arrays"}


def test_primitive_cell_owns_a_normalized_read_only_snapshot() -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_scaled_positions(atoms.get_scaled_positions() + [2.0, -1.0, 0.0])
    primitive = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 0.01})
    atoms.positions[:] = 0.0

    assert np.all((primitive.scaled_positions >= 0.0) & (primitive.scaled_positions < 1.0))
    assert not primitive.cell.flags.writeable
    assert not primitive.scaled_positions.flags.writeable
    assert not primitive.atomic_numbers.flags.writeable
    assert not np.allclose(primitive.cartesian_positions, 0.0)


def test_primitive_cell_refuses_nonperiodic_or_singular_input() -> None:
    nonperiodic = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=False)
    singular = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.zeros((3, 3)), pbc=True)

    with pytest.raises(ValueError, match="periodic"):
        ClusterSpace(nonperiodic, symprec=1e-05, cutoffs={2: 0.01})
    with pytest.raises(ValueError, match="nonsingular"):
        ClusterSpace(singular, symprec=1e-05, cutoffs={2: 0.01})

    repeated = bulk("Si", "diamond", a=5.43).repeat((2, 1, 1))
    with pytest.raises(ValueError, match="primitive cell contains"):
        ClusterSpace(repeated, symprec=1e-05, cutoffs={2: 0.01})


def test_primitive_symmetry_is_a_closed_action_on_the_motif() -> None:
    primitive = ClusterSpace(bulk("Si", "diamond", a=5.43), symprec=1e-05, cutoffs={2: 0.01})
    symmetry = discover_symmetry(
        primitive.cell, primitive.scaled_positions, primitive.atomic_numbers, primitive.symprec
    )

    assert symmetry.size > 1
    np.testing.assert_array_equal(
        np.sort(symmetry.site_permutations, axis=1),
        np.broadcast_to(np.arange(primitive.n_atoms), symmetry.site_permutations.shape),
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
    primitive = ClusterSpace(bulk("Ar", "sc", a=1.0), symprec=1e-05, cutoffs={2: 0.01})
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
        assert (
            discover_symmetry(
                primitive.cell,
                primitive.scaled_positions,
                primitive.atomic_numbers,
                primitive.symprec,
            ).size
            == 1
        )
    else:
        with pytest.raises(ValueError, match="0 sites within symprec"):
            discover_symmetry(
                primitive.cell,
                primitive.scaled_positions,
                primitive.atomic_numbers,
                primitive.symprec,
            )


def test_exact_algebra_handles_large_integers_and_certifies_a_kernel() -> None:
    scale = 2**40
    with pytest.raises(OverflowError):
        exact_product(
            np.asarray([[scale, scale]], dtype=np.int64),
            np.asarray([[scale], [scale]], dtype=np.int64),
        )
    np.testing.assert_array_equal(exact_product([[2, 3]], [[4], [5]]), [[23]])

    matrix = np.asarray([[1, -2, 1], [2, 1, 1]], dtype=np.int64)
    kernel = exact_kernel(matrix)
    assert exact_rank(matrix) == 2
    assert kernel.dtype == np.int64
    verify_kernel(matrix, kernel)
    assert kernel.shape[1] == matrix.shape[1] - exact_rank(matrix)

    with pytest.raises(OverflowError):
        exact_kernel([[2**70, 1]])
    with pytest.raises(ValueError, match="declared integers"):
        exact_rank(np.asarray([[1.0]]))


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
    with pytest.raises(OverflowError):
        rotate_q_labels(np.asarray([scale, 0, 0], dtype=np.int64), rotation)
    np.testing.assert_array_equal(
        rotate_q_labels([scale, 0, 0], rotation, denominator=7),
        np.asarray([scale % 7, -(scale**2) % 7, 0]),
    )
