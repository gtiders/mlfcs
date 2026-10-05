"""Numerical behavior of geometry."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from mlfcs import ClusterSpace
from mlfcs.geometry.primitive import LatticeSite
from mlfcs.geometry.symmetry import discover_symmetry
from mlfcs.tensors import apply_tensor_action


@pytest.mark.parametrize("order", (2, 3, 4, 5), ids=("FC2", "FC3", "FC4", "FC5"))
def test_lattice_tensor_action_matches_numpy_contraction_and_permutation(order):
    """Verify lattice tensor action matches numpy contraction and permutation."""
    rng = np.random.default_rng(71)
    rotation = np.array([[1, 7, 0], [0, 1, 0], [0, 0, -1]], dtype=np.int64)
    values = rng.integers(-3, 4, (3**order, 3), dtype=np.int64)
    permutation = tuple(reversed(range(order)))
    transformed = values.T.reshape((3,) + (3,) * order)
    for axis in range(order):
        transformed = np.tensordot(rotation, transformed, axes=((1,), (axis + 1,)))
        transformed = np.moveaxis(transformed, 0, axis + 1)
    expected = np.transpose(transformed, (0,) + tuple(i + 1 for i in permutation)).reshape(3, -1).T
    np.testing.assert_array_equal(
        apply_tensor_action(values, rotation, np.asarray(permutation, dtype=np.int64)), expected
    )


def test_primitive_symmetry_is_a_closed_action_on_the_motif() -> None:
    """Verify primitive symmetry is a closed action on the motif."""
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


def test_periodic_matching_enumerates_all_images_inside_tolerance():
    """Verify periodic matching enumerates all images inside tolerance."""
    from mlfcs.geometry.periodic import PeriodicGeometry

    geometry = PeriodicGeometry(np.eye(3))
    candidates, shifts = geometry.matching_images(
        np.asarray([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]), tolerance=0.6
    )
    assert candidates.tolist() == [0, 0, 1]
    assert {tuple(row) for row in shifts[:2]} == {(-1, 0, 0), (0, 0, 0)}
    np.testing.assert_array_equal(shifts[2], [0, 0, 0])


def test_periodic_matching_uses_strict_radius_and_handles_no_candidates():
    """Verify periodic matching uses strict radius and handles no candidates."""
    from mlfcs.geometry.periodic import PeriodicGeometry

    geometry = PeriodicGeometry(np.eye(3))
    for vectors in (np.empty((0, 3)), np.asarray([[0.5, 0.0, 0.0]])):
        candidates, shifts = geometry.matching_images(vectors, tolerance=0.5)
        assert candidates.shape == (0,)
        assert shifts.shape == (0, 3)
