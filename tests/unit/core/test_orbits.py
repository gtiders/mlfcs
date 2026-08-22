import numpy as np
from ase import Atoms

from mlfcs.core.geometry import make_supercell
from mlfcs.core.orbits import (
    _compatible_sorted_tails,
    _joint_periodic_cluster_geometry,
    _label_symmetric_basis,
    tensor_action_matrix,
)


def test_identity_tensor_action():
    action = tensor_action_matrix(np.eye(3), (0, 1, 2), 3)
    np.testing.assert_allclose(action, np.eye(27))


def test_periodic_triangle_requires_compatible_joint_images():
    primitive = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)
    supercell, _ = make_supercell(primitive, (3, 1, 1))
    distances, compatible = _joint_periodic_cluster_geometry(supercell, 1, 1.1)

    # Every pair has an independent MIC distance of 1, but atoms 1 and 2
    # occupy opposite minimum images relative to anchor 0. They cannot form
    # one periodic triangle within the 1.1 cutoff.
    np.testing.assert_allclose(distances[0], [0, 1, 1])
    assert not compatible[0, 1, 2]


def test_sorted_tail_generator_prunes_incompatible_prefixes():
    compatibility = np.ones((4, 4), dtype=bool)
    compatibility[1, 3] = compatibility[3, 1] = False
    actual = list(_compatible_sorted_tails([0, 1, 2, 3], 3, compatibility))
    expected = [
        tail
        for tail in np.ndindex((4, 4, 4))
        if tuple(sorted(tail)) == tail
        and all(compatibility[left, right] for left in tail for right in tail)
    ]
    assert actual == expected
    assert all(not ({1, 3} <= set(tail)) for tail in actual)


def test_sixth_order_onsite_label_basis_has_28_components():
    basis = _label_symmetric_basis((0,) * 6)
    assert basis.shape == (3**6, 28)
    np.testing.assert_allclose(basis.T @ basis, np.eye(28), atol=1e-14)
