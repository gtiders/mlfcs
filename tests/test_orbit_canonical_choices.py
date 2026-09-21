r"""Canonical choices of the orbit algebra: observation rows and exact lattice bases.

Both choices under test here are supposed to depend on the *subspace* the orbit spans and
not on the accident of how it was written down, which is what a reviewer can check without
re-deriving the algebra:

* the observation rows are selected from an orthonormal basis $Q$, so they must survive a
  right multiplication $Q \\to Q O$ with any orthogonal $O$ -- the volumes the greedy
  maximizes are $\det(Q[S] O O^T Q[S]^T) = \det(Q[S] Q[S]^T)$;
* the exact lattice basis is the column Hermite normal form of the invariant kernel, so an
  atom-permuted primitive cell of the same crystal, and any unimodular column change of the
  kernel, must leave the stored integers bit for bit identical.

The materials are the four the branch already uses elsewhere: a simple cubic cell, the
primitive diamond fcc cell, the hexagonal hcp cell and a tilted triclinic two-atom cell.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlfcs.interactions.algebra import rendering
from mlfcs.interactions.algebra.exact import canonical_lattice_basis
from mlfcs.interactions.algebra.rendering import (
    _MAX_OBSERVATION_CONDITION,
    observation_condition,
    select_observation_rows,
)
from mlfcs.interactions.primitive.builder import build_primitive_interaction_space

_SYMPREC = 1e-5
_ORDERS = (2, 3, 4)


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


def _space(atoms: Atoms, order: int, cutoff: float):
    return build_primitive_interaction_space(
        atoms,
        order=order,
        cutoff=cutoff,
        max_body_order=None,
        symprec=_SYMPREC,
    )


def _permuted(atoms: Atoms) -> Atoms:
    """Return the same crystal with its atoms written in the reverse order."""
    order = np.arange(len(atoms))[::-1]
    return Atoms(
        atoms.numbers[order],
        scaled_positions=atoms.get_scaled_positions(wrap=False)[order],
        cell=np.asarray(atoms.cell),
        pbc=True,
    )


def _bounded_unimodular(size: int, rng) -> np.ndarray:
    """A small-entry unimodular matrix: column order, column signs and two unit shears.

    A change of parameter basis must be invisible to the canonical form, and any unimodular
    matrix is such a change.  The product below keeps ``abs(det) == 1`` while its entries
    stay of order one, which keeps the exact Hermite normal form of an 81-component orbit
    basis fast; the aggressive random shears are exercised in ``test_exact_rank_kernel``
    where the matrices are tiny.
    """
    permutation = np.eye(size, dtype=np.int64)[:, rng.permutation(size)]
    signs = np.diag(np.where(rng.random(size) < 0.5, -1, 1).astype(np.int64))
    shears = np.eye(size, dtype=np.int64)
    for _ in range(2):
        left, right = (int(value) for value in rng.integers(0, size, 2))
        if left != right:
            shears[:, left] += shears[:, right]
    return shears @ signs @ permutation


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("material", sorted(_MATERIALS))
def test_observation_rows_depend_on_the_subspace_alone(material, order):
    factory, cutoff = _MATERIALS[material]
    space = _space(factory(), order, cutoff)
    rng = np.random.default_rng(20240921 + order)

    assert space.orbits
    for orbit in space.orbits:
        basis = orbit.cartesian_basis
        expected = select_observation_rows(basis, orbit.dimension)

        assert expected.tolist() == sorted(expected.tolist())
        assert expected.tolist() == orbit.observation_rows.tolist()
        for reflection in (False, True):
            orthogonal = np.linalg.qr(rng.normal(size=(orbit.dimension, orbit.dimension)))[0]
            if reflection:  # det = -1, which the volume argument has to tolerate as well
                orthogonal[:, 0] *= -1.0
            other = select_observation_rows(basis @ orthogonal, orbit.dimension)
            assert other.tolist() == expected.tolist()


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("material", sorted(_MATERIALS))
def test_orbit_algebra_survives_an_atom_permutation(material, order):
    factory, cutoff = _MATERIALS[material]
    atoms = factory()
    direct = _space(atoms, order, cutoff)
    permuted = _space(_permuted(atoms), order, cutoff)

    assert len(permuted.orbits) == len(direct.orbits)
    for left, right in zip(direct.orbits, permuted.orbits, strict=True):
        assert left.observation_rows.tolist() == right.observation_rows.tolist()
        assert left.observation_condition == right.observation_condition
        np.testing.assert_array_equal(left.exact_lattice_basis, right.exact_lattice_basis)


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("material", sorted(_MATERIALS))
def test_observation_condition_is_the_stored_block_and_within_the_limit(material, order):
    factory, cutoff = _MATERIALS[material]
    space = _space(factory(), order, cutoff)

    assert space.orbits
    for orbit in space.orbits:
        block = orbit.cartesian_basis[orbit.observation_rows]
        assert block.shape == (orbit.dimension, orbit.dimension)
        assert observation_condition(block) == pytest.approx(orbit.observation_condition, rel=1e-12)
        assert 1.0 <= orbit.observation_condition < _MAX_OBSERVATION_CONDITION


def test_observation_rows_reject_a_non_orthonormal_basis():
    # The raw Cartesian basis of an orbit is not orthonormal, and the volumes the greedy
    # maximizes mean nothing for it: the caller has to pass the QR factor.
    with pytest.raises(ValueError, match="orthonormal"):
        select_observation_rows(np.ones((4, 2)), 2)
    with pytest.raises(ValueError, match=r"max\(abs\(Q\.T @ Q - I\)\)"):
        select_observation_rows(np.eye(4)[:, :2] * np.asarray([1.0, 1.001]), 2)

    # A basis that is orthonormal to the documented tolerance is accepted.
    basis = np.eye(4)[:, :2]
    basis[:, 1] = basis[:, 1] + 1e-9 * basis[:, 0]
    basis[:, 1] /= np.linalg.norm(basis[:, 1])
    assert select_observation_rows(basis, 2).tolist() == [0, 1]


def test_observation_rows_refuse_an_ill_conditioned_selection(monkeypatch):
    # The greedy never came close to the documented limit on any material of this branch:
    # an orthonormal basis is a tight frame, so the residual the greedy maximizes at step
    # ``j`` is at least ``sqrt((d - j + 1) / rows)`` and the selected block stays well
    # conditioned (the materials above reach 4.9).  The threshold is therefore driven
    # directly here, which is the only way to exercise it without a basis the
    # orthonormality check has to reject first.
    space = _space(_diamond_silicon(), 3, -1)
    orbit = space.orbits[-1]
    assert select_observation_rows(orbit.cartesian_basis, orbit.dimension).tolist() == (
        orbit.observation_rows.tolist()
    )

    monkeypatch.setattr(rendering, "_MAX_OBSERVATION_CONDITION", 1.0)
    with pytest.raises(
        RuntimeError, match=r"condition number .* above the accepted maximum 1e\+00"
    ):
        select_observation_rows(orbit.cartesian_basis, orbit.dimension)


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("material", sorted(_MATERIALS))
def test_exact_lattice_basis_ignores_column_sign_and_order(material, order):
    factory, cutoff = _MATERIALS[material]
    space = _space(factory(), order, cutoff)
    rng = np.random.default_rng(20240922 + order)

    assert space.orbits
    for orbit in space.orbits:
        basis = orbit.exact_lattice_basis

        assert basis.dtype == np.int64
        expected = canonical_lattice_basis(basis)
        # The canonical form is idempotent; the orbit stores the kernel basis mapped into
        # the component space, which is a generating set of the same lattice but not
        # necessarily that lattice's own Hermite normal form.
        np.testing.assert_array_equal(canonical_lattice_basis(expected), expected)
        if orbit.dimension == 0:
            continue
        for _ in range(3):
            changed = basis @ _bounded_unimodular(orbit.dimension, rng)
            np.testing.assert_array_equal(canonical_lattice_basis(changed), expected)
