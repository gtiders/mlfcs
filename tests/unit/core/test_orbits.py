import numpy as np
from ase import Atoms

from mlfcs.core.geometry import make_supercell
from mlfcs.core.orbits import (
    _joint_periodic_cluster_geometry,
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
