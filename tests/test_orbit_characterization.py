"""Characterize the orbit algebra every implementation has to preserve.

These tests describe the *physics* of a primitive orbit space, independently of how a
basis is stored:

* how many orbits exist and how many independent tensor parameters each one carries;
* that each orbit's basis is invariant under every space-group operation that maps the
  anchored cluster onto itself, which the tests enumerate themselves in the source
  lattice frame;
* that the Cartesian subspace an orbit spans does not change when the same crystal is
  expressed in a different unimodular basis.

A basis is only defined up to an invertible change of coordinates, so subspaces are
compared through the projector ``Q Q^T`` and never column by column.  The stabilizer
enumeration is a necessary condition -- it can never prove that a parameterization is
minimal, but a frame, rotation or axis-permutation mistake fails it immediately.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlfcs.interactions.algebra.actions import TensorAction
from mlfcs.interactions.primitive.builder import build_primitive_interaction_space
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

# A genuine unimodular change of the primitive basis: the same crystal written with a
# different set of integer translations, which every orbit statement has to survive.
_SHEAR = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)

_SYMPREC = 1e-5


def _cubic_silicon() -> Atoms:
    return Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)


def _diamond_silicon() -> Atoms:
    return bulk("Si", "diamond", a=5.43)


def _hexagonal_magnesium() -> Atoms:
    return bulk("Mg", "hcp", a=3.2, c=5.2)


def _tilted_binary() -> Atoms:
    cell = np.asarray([[3.2, 0.0, 0.0], [1.1, 3.0, 0.0], [0.7, 0.9, 4.5]])
    return Atoms(
        "SiGe",
        scaled_positions=[[0.0, 0.0, 0.0], [0.35, 0.45, 0.55]],
        cell=cell,
        pbc=True,
    )


_MATERIALS = {
    "cubic-silicon": (_cubic_silicon, 4.1),
    "diamond-silicon": (_diamond_silicon, -1),
    "hexagonal-magnesium": (_hexagonal_magnesium, -1),
    "tilted-binary": (_tilted_binary, -1),
}

_ORBIT_COUNTS = {
    "cubic-silicon": {2: 2, 3: 1},
    "diamond-silicon": {2: 2, 3: 2},
    "hexagonal-magnesium": {2: 2, 3: 2},
    "tilted-binary": {2: 3, 3: 4},
}


def _sheared(atoms: Atoms, matrix: np.ndarray) -> Atoms:
    """Return the same crystal expressed in the primitive basis ``matrix @ cell``."""
    cell = np.asarray(atoms.cell)
    scaled = np.mod(atoms.get_scaled_positions(wrap=True) @ np.linalg.inv(matrix), 1.0)
    return Atoms(atoms.numbers, scaled_positions=scaled, cell=matrix @ cell, pbc=True)


def _orbit_basis(orbit) -> np.ndarray:
    """Return the Cartesian column basis that carries one orbit's parameters."""
    return orbit.cartesian_basis


def _projector(basis: np.ndarray) -> np.ndarray:
    """Return the orthogonal projector onto the span of ``basis``."""
    orthonormal, _ = np.linalg.qr(np.asarray(basis, dtype=float))
    return orthonormal @ orthonormal.T


def _build(atoms: Atoms, order: int, cutoff: float):
    return build_primitive_interaction_space(
        atoms,
        order=order,
        cutoff=cutoff,
        max_body_order=None,
        symprec=_SYMPREC,
    )


def _anchored_stabilizers(atoms: Atoms, representative, order: int) -> list[TensorAction]:
    """Enumerate the group elements that map one anchored cluster onto itself.

    The enumeration is independent of the orbit algebra: it applies spglib operations and
    IFC axis permutations in the *source* frame to the anchored labels and keeps the
    combinations whose image is the same anchored label multiset.  Each surviving element
    acts on tensor components as its Cartesian rotation followed by its axis permutation,
    which is the operation an invariant parameterization has to satisfy.
    """
    symmetry = PrimitiveSymmetryOperations.from_atoms(atoms, symprec=_SYMPREC)
    labels = np.asarray(representative.labels, dtype=np.int64)
    rows = [tuple(int(value) for value in row) for row in labels.tolist()]
    actions = []
    for operation in range(symmetry.size):
        sites = symmetry.site_permutations[operation][labels[:, 0]]
        shifts = symmetry.site_shifts[operation][labels[:, 0]]
        translated = labels[:, 1:] @ symmetry.rotations[operation].T + shifts
        moved = np.column_stack((sites, translated))
        rows_moved = [tuple(int(value) for value in row) for row in moved.tolist()]
        rotation = np.asarray(symmetry.cartesian_rotations[operation].T, dtype=float)
        for swap in itertools.permutations(range(order)):
            # The traversal permutes the cluster axes first and re-anchors afterwards, and
            # it identifies states by their *ordered* rows: an element is a stabilizer only
            # when the permuted row tuple is exactly the one it started from.
            images = np.asarray([rows_moved[index] for index in swap], dtype=np.int64)
            images[:, 1:] -= images[0, 1:]
            if [tuple(int(value) for value in row) for row in images.tolist()] != rows:
                continue
            actions.append(TensorAction(rotation, tuple(swap), order))
    return actions


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("material", sorted(_MATERIALS))
def test_orbit_dimensions_match_the_crystal_symmetry(material, order):
    factory, cutoff = _MATERIALS[material]
    space = _build(factory(), order, cutoff)

    assert space.order == order
    assert space.n_parameters == sum(orbit.dimension for orbit in space.orbits)
    assert space.n_parameters > 0
    for orbit in space.orbits:
        basis = _orbit_basis(orbit)
        assert basis.shape == (3**order, orbit.dimension)
        projector = _projector(basis)
        np.testing.assert_allclose(projector @ projector, projector, rtol=1e-10, atol=1e-10)

    assert len(space.orbits) == _ORBIT_COUNTS[material][order]


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("material", sorted(_MATERIALS))
def test_orbit_basis_is_invariant_under_the_anchored_cluster_stabilizers(material, order):
    factory, cutoff = _MATERIALS[material]
    atoms = factory()
    space = _build(atoms, order, cutoff)

    for orbit in space.orbits:
        basis = _orbit_basis(orbit)
        stabilizers = _anchored_stabilizers(atoms, orbit.representative, order)
        assert stabilizers, "the identity always stabilizes a cluster"
        for action in stabilizers:
            np.testing.assert_allclose(
                action.apply_columns(basis),
                basis,
                rtol=1e-8,
                atol=1e-8,
                err_msg=f"{material} order {order}: basis is not stabilizer invariant",
            )


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("material", sorted(_MATERIALS))
def test_orbit_subspaces_are_stable_under_a_unimodular_change_of_basis(material, order):
    factory, cutoff = _MATERIALS[material]
    atoms = factory()
    direct = _build(atoms, order, cutoff)
    sheared = _build(_sheared(atoms, _SHEAR), order, cutoff)

    assert len(sheared.orbits) == len(direct.orbits)
    assert sheared.n_parameters == direct.n_parameters
    assert [orbit.dimension for orbit in sheared.orbits] == [
        orbit.dimension for orbit in direct.orbits
    ]

    remaining = [_projector(_orbit_basis(orbit)) for orbit in sheared.orbits]
    for orbit in direct.orbits:
        target = _projector(_orbit_basis(orbit))
        matches = [
            index
            for index, candidate in enumerate(remaining)
            if np.allclose(candidate, target, rtol=1e-8, atol=1e-8)
        ]
        assert matches, f"{material} order {order}: no sheared orbit spans the same subspace"
        remaining.pop(matches[0])
