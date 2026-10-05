"""Numerical behavior of cluster space."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlfcs.cluster_space import Cluster, ClusterSpace
from mlfcs.cluster_space.candidates import enumerate_candidate_labels
from mlfcs.cluster_space.construction import _axis_permutations
from mlfcs.cluster_space.orbits import compute_orbit_actions
from mlfcs.geometry.symmetry import discover_symmetry


def test_orbit_actions_produce_distinct_cluster_images() -> None:
    """Verify orbit actions produce distinct cluster images."""
    primitive = ClusterSpace(bulk("Si", "diamond", a=5.43), symprec=1e-05, cutoffs={2: 0.01})
    symmetry = discover_symmetry(
        primitive.cell, primitive.scaled_positions, primitive.atomic_numbers, primitive.symprec
    )
    candidate = Cluster.from_labels(
        enumerate_candidate_labels(
            primitive.cell, primitive.scaled_positions, order=3, cutoff=4.0, max_body_order=3
        )[0]
    )
    values, array = _axis_permutations(3)
    representative, clusters, _operations, _permutations, stabilizers = compute_orbit_actions(
        candidate, symmetry, values, array
    )
    assert representative == candidate
    assert candidate in clusters
    keys = tuple(tuple(value for row in cluster.labels for value in row) for cluster in clusters)
    assert len(keys) == len(set(keys))
    assert stabilizers


@pytest.mark.parametrize(
    ("atoms", "order", "expected_orbits", "expected_parameters"),
    (
        (bulk("Si", "diamond", a=5.43), 2, 3, 7),
        (bulk("Si", "diamond", a=5.43), 3, 6, 36),
        (bulk("Mg", "hcp", a=3.2, c=5.2), 3, 6, 42),
    ),
    ids=("Si-FC2", "Si-FC3", "Mg-FC3"),
)
def test_cluster_space_has_the_characterized_physical_dimension(
    atoms: Atoms, order: int, expected_orbits: int, expected_parameters: int
) -> None:
    """Verify cluster space has the characterized physical dimension."""
    primitive = atoms
    space = ClusterSpace(
        primitive, cutoffs={order: 4.0}, max_body_orders={order: order}, symprec=1e-05
    )
    assert len(space.orbits) == expected_orbits
    assert space.n_parameters == expected_parameters
    for orbit in space.orbits:
        np.testing.assert_allclose(
            orbit.observation_matrix, np.eye(orbit.dimension), atol=2e-14, rtol=0.0
        )


def test_lattice_shear_preserves_orbit_dimensions() -> None:
    """Verify lattice shear preserves orbit dimensions."""
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
    """Verify one space owns a stable multi order parameter layout."""
    primitive = bulk("Si", "diamond", a=5.43)
    space = ClusterSpace(
        primitive, cutoffs={3: 4.0, 2: 4.0}, max_body_orders={2: 2, 3: 3}, symprec=1e-05
    )
    assert space.orders == (2, 3)
    assert space.block(2).parameters == slice(0, 7)
    assert space.block(3).parameters == slice(7, 43)
    assert space.n_parameters == 43
    assert all(orbit.representative.order == 2 for orbit in space.orbits[space.block(2).orbits])
    assert all(orbit.representative.order == 3 for orbit in space.orbits[space.block(3).orbits])


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


def test_primitive_snapshot_preserves_wrapped_geometry() -> None:
    """Verify primitive snapshot preserves wrapped geometry."""
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_scaled_positions(atoms.get_scaled_positions() + [2.0, -1.0, 0.0])
    primitive = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 0.01})
    atoms.positions[:] = 0.0
    assert np.all((primitive.scaled_positions >= 0.0) & (primitive.scaled_positions < 1.0))
    assert not primitive.cell.flags.writeable
    assert not primitive.scaled_positions.flags.writeable
    assert not primitive.atomic_numbers.flags.writeable
    assert not np.allclose(primitive.cartesian_positions, 0.0)


@pytest.mark.reference
@pytest.mark.parametrize(
    "reference",
    json.loads((Path(__file__).parent / "data" / "symmetry_reference.json").read_text()),
    ids=lambda record: f"{record['material']}-FC{record['order']}",
)
def test_orbits_match_saved_symmetry_reference(reference):
    """Verify orbits match saved symmetry reference."""
    sympy = pytest.importorskip("sympy")
    from sympy.matrices.normalforms import hermite_normal_form

    material, order, cutoff = (reference["material"], reference["order"], reference["cutoff"])
    atoms = bulk("Si", "diamond", a=5.43) if material == "Si" else bulk("Mg", "hcp", a=3.2, c=5.2)
    primitive = atoms
    space = ClusterSpace(
        primitive, cutoffs={order: cutoff}, max_body_orders={order: order}, symprec=1e-05
    )
    assert len(space.orbits) == len(reference["orbits"])
    assert space.n_parameters == reference["n_parameters"]
    for index, orbit in enumerate(space.orbits):
        old = reference["orbits"][index]
        np.testing.assert_array_equal(orbit.representative.labels, old["representative"])
        np.testing.assert_array_equal([c.labels for c in orbit.clusters], old["images"])
        np.testing.assert_array_equal(orbit.operations, old["operations"])
        np.testing.assert_array_equal(orbit.permutations, old["permutations"])
        assert hermite_normal_form(
            sympy.Matrix(orbit.lattice_basis.tolist())
        ) == hermite_normal_form(sympy.Matrix(old["lattice_basis"]))
        old_basis = np.asarray(old["component_basis"], dtype=float)
        q1, _ = np.linalg.qr(orbit.component_basis)
        q2, _ = np.linalg.qr(old_basis)
        np.testing.assert_allclose(q1 @ q1.T, q2 @ q2.T, atol=2e-12)
