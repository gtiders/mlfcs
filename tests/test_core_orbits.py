import numpy as np
from ase import Atoms

from mlfcs.clusters.orbits import (
    _canonical_cluster,
    _compatible_sorted_tails,
    _joint_periodic_cluster_geometry,
    _label_symmetric_basis,
    build_orbit_space,
    tensor_action_matrix,
)
from mlfcs.core.symmetry import SymmetryOperations
from mlfcs.structure.geometry import make_supercell


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


def test_max_body_order_filters_clusters_by_number_of_distinct_sites():
    primitive = Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, index = make_supercell(primitive, (1, 1, 1))
    symmetry = SymmetryOperations.from_atoms(primitive, supercell)
    unrestricted = build_orbit_space(supercell, index, symmetry, order=3, cutoff=2.0)
    pair_only = build_orbit_space(supercell, index, symmetry, order=3, cutoff=2.0, max_body_order=1)

    assert any(len(set(orbit.representative)) == 2 for orbit in unrestricted.orbits)
    assert pair_only.orbits
    assert all(len(set(orbit.representative)) == 1 for orbit in pair_only.orbits)


def test_omitted_body_limit_preserves_the_unrestricted_orbit_space():
    primitive = Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, index = make_supercell(primitive, (1, 1, 1))
    symmetry = SymmetryOperations.from_atoms(primitive, supercell)
    omitted = build_orbit_space(supercell, index, symmetry, order=3, cutoff=2.0)
    explicit = build_orbit_space(
        supercell, index, symmetry, order=3, cutoff=2.0, max_body_order=None
    )

    assert omitted.displacement_keys == explicit.displacement_keys
    assert [orbit.representative for orbit in omitted.orbits] == [
        orbit.representative for orbit in explicit.orbits
    ]


def test_canonical_orbit_is_kept_when_only_a_noncanonical_seed_passes_support():
    """Periodic support discovery must not require the canonical image as a seed."""
    primitive = Atoms(
        "H2",
        scaled_positions=[[0, 0, 0], [0.49, 0.49, 0]],
        cell=[[2.0, 0, 0], [1.8, 0.7, 0], [0, 0, 4.0]],
        pbc=True,
    )
    supercell, index = make_supercell(primitive, (2, 2, 1))
    symmetry = SymmetryOperations.from_atoms(primitive, supercell, symprec=1e-7)
    space = build_orbit_space(supercell, index, symmetry, order=3, cutoff=1.0)

    representatives = {orbit.representative for orbit in space.orbits}
    distances, compatibility = _joint_periodic_cluster_geometry(supercell, 2, 1.0)
    canonical_from_seeds = set()
    noncanonical_seeds = 0
    for first in range(2):
        neighbors = np.flatnonzero(distances[first] < 1.0).tolist()
        for tail in _compatible_sorted_tails(neighbors, 2, compatibility[first]):
            seed = (first, *tail)
            canonical = _canonical_cluster(seed, index, symmetry)
            canonical_from_seeds.add(canonical)
            noncanonical_seeds += canonical != seed

    assert noncanonical_seeds > 0
    assert canonical_from_seeds <= representatives
