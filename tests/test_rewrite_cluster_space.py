"""Small, physical contracts for the isolated primitive cluster-space rewrite."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase import Atoms
from ase.build import bulk

from mlfcs.cluster_space import Cluster, ClusterSpace
from mlfcs.cluster_space._orbits import _integer_boundary, _orbit_actions
from mlfcs.cluster_space.builder import _axis_permutations
from mlfcs.cluster_space.candidates import _candidate_labels
from mlfcs.core import LatticeSite
from mlfcs.core.symmetry import discover_symmetry


def test_cluster_space_depends_only_on_the_rewritten_core() -> None:
    assert internal_dependencies("cluster_space") == {"core", "_arrays", "algebra"}


def test_first_uncovered_candidate_is_the_orbit_representative() -> None:
    primitive = ClusterSpace(bulk("Si", "diamond", a=5.43), symprec=1e-05, cutoffs={2: 0.01})
    symmetry = discover_symmetry(
        primitive.cell, primitive.scaled_positions, primitive.atomic_numbers, primitive.symprec
    )
    candidate = Cluster.from_labels(
        _candidate_labels(
            primitive.cell,
            primitive.scaled_positions,
            order=3,
            cutoff=4.0,
            max_body_order=3,
        )[0]
    )
    values, array = _axis_permutations(3)

    representative, clusters, _operations, _permutations, stabilizers, keys = _orbit_actions(
        candidate, symmetry, values, array
    )

    assert representative == candidate
    assert candidate in clusters
    assert len(keys) == len(set(keys))
    assert stabilizers


@pytest.mark.parametrize(
    ("atoms", "order", "expected_orbits", "expected_parameters"),
    (
        (bulk("Si", "diamond", a=5.43), 2, 3, 7),
        (bulk("Si", "diamond", a=5.43), 3, 6, 36),
        (bulk("Mg", "hcp", a=3.2, c=5.2), 3, 6, 42),
    ),
)
def test_cluster_space_has_the_characterized_physical_dimension(
    atoms: Atoms, order: int, expected_orbits: int, expected_parameters: int
) -> None:
    primitive = atoms
    space = ClusterSpace(
        primitive,
        cutoffs={order: 4.0},
        max_body_orders={order: order},
        symprec=1e-05,
    )
    assert len(space.orbits) == expected_orbits
    assert space.n_parameters == expected_parameters
    for orbit in space.orbits:
        np.testing.assert_allclose(
            orbit.observation_matrix,
            np.eye(orbit.dimension),
            atol=2e-14,
            rtol=0.0,
        )


def test_integer_shear_preserves_orbit_and_parameter_dimensions() -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    change = np.asarray([[1, 7, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    inverse = np.asarray([[1, -7, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    sheared = Atoms(
        numbers=atoms.numbers,
        scaled_positions=np.mod(atoms.get_scaled_positions(wrap=False) @ inverse, 1.0),
        cell=change @ np.asarray(atoms.cell),
        pbc=True,
    )

    spaces = []
    for structure in (atoms, sheared):
        primitive = structure
        spaces.append(
            ClusterSpace(primitive, cutoffs={3: 4.0}, max_body_orders={3: 3}, symprec=1e-05)
        )

    assert [orbit.dimension for orbit in spaces[0].orbits] == [
        orbit.dimension for orbit in spaces[1].orbits
    ]
    assert spaces[0].n_parameters == spaces[1].n_parameters


def test_one_space_owns_a_stable_multi_order_parameter_layout() -> None:
    primitive = bulk("Si", "diamond", a=5.43)
    space = ClusterSpace(
        primitive,
        cutoffs={3: 4.0, 2: 4.0},
        max_body_orders={2: 2, 3: 3},
        symprec=1e-05,
    )

    assert space.orders == (2, 3)
    assert space.block(2).parameters == slice(0, 7)
    assert space.block(3).parameters == slice(7, 43)
    assert space.n_parameters == 43
    assert all(orbit.representative.order == 2 for orbit in space.orbits[space.block(2).orbits])
    assert all(orbit.representative.order == 3 for orbit in space.orbits[space.block(3).orbits])


def test_observation_selection_contains_no_model_deciding_float_threshold() -> None:
    path = Path("src/mlfcs/cluster_space/basis.py")
    tree = ast.parse(path.read_text())
    comparisons = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)]
    literals = {
        node.value
        for comparison in comparisons
        for node in ast.walk(comparison)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    }
    assert literals == {0.0}


def test_compiled_orbit_boundary_rejects_only_a_proven_integer_overflow() -> None:
    primitive = ClusterSpace(bulk("Ar", "sc", a=1.0), symprec=1e-05, cutoffs={2: 0.01})
    symmetry = discover_symmetry(
        primitive.cell, primitive.scaled_positions, primitive.atomic_numbers, primitive.symprec
    )
    first_overflowing_translation = np.iinfo(np.int64).max // 2 + 1
    cluster = Cluster(
        (
            LatticeSite(0),
            LatticeSite(0, (first_overflowing_translation, 0, 0)),
        )
    )

    with pytest.raises(OverflowError, match="proven bound"):
        _integer_boundary(cluster.labels, symmetry)


@pytest.mark.parametrize(
    "fractional", [[-0.2, 1.3, 2.0], [-1e-300, 1.0, 0.5], [1.0 - 1e-15, -2.0, 0.3]]
)
def test_primitive_positions_use_ase_wrap_without_mutating_input(fractional):
    """Match ASE wrapping at periodic boundaries while preserving the caller's atoms."""
    atoms = bulk("Ar", "sc", a=1.0)
    atoms.set_scaled_positions([fractional])
    atoms.set_masses([41.0])
    original = atoms.positions.copy()
    expected = atoms.get_scaled_positions(wrap=True)
    space = ClusterSpace(atoms, cutoffs={2: 0.01})
    np.testing.assert_array_equal(space.scaled_positions, expected)
    np.testing.assert_array_equal(atoms.positions, original)
    np.testing.assert_array_equal(space.masses, [41.0])
    assert np.all(space.scaled_positions >= 0) and np.all(space.scaled_positions < 1)
