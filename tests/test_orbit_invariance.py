"""Unimodular invariance and conditioning of the lattice-frame orbit algebra.

An integer orbit algebra is only useful if it does not depend on how the crystal is
written down.  These tests express the *same* crystal in several unimodular primitive
bases, including a strongly sheared one that used to blow the integer tensor
coefficients out of ``int64`` at third order, and require the orbit algebra to agree
exactly:

* the same orbits with the same invariant dimensions;
* the same observation condition numbers, without the shear degrading them;
* the same Cartesian invariant subspaces;
* the same physical force constants after expansion;
* the same finite-difference reconstruction, which exercises the displacement plan, the
  observation solve and the expansion together.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones

from mlfcs import FiniteDifferenceCalculation, build_supercell
from mlfcs.force_constants.expansion import expand_primitive_parameters
from mlfcs.interactions.primitive.builder import build_primitive_interaction_space
from mlfcs.interactions.realization import realize_interaction_space

# Every shear is a unimodular integer matrix (unit diagonal, so determinant one) that
# keeps the crystal and changes nothing but the primitive basis.
_SHEARS = {
    "adjacent": np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64),
    "crossed": np.asarray([[1, -2, 1], [0, 1, -1], [0, 0, 1]], dtype=np.int64),
    "large": np.asarray([[1, 7, 5], [0, 1, 3], [0, 0, 1]], dtype=np.int64),
}


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


def _rebased(atoms: Atoms, matrix: np.ndarray) -> Atoms:
    """Return the same crystal written in the primitive basis ``matrix @ cell``."""
    cell = np.asarray(atoms.cell)
    scaled = np.mod(atoms.get_scaled_positions(wrap=True) @ np.linalg.inv(matrix), 1.0)
    return Atoms(atoms.numbers, scaled_positions=scaled, cell=matrix @ cell, pbc=True)


def _space(atoms: Atoms, order: int, cutoff: float):
    return build_primitive_interaction_space(
        atoms,
        order=order,
        cutoff=cutoff,
        max_body_order=None,
        symprec=1e-5,
    )


def _physical_clusters(atoms: Atoms, sparse) -> dict[tuple, np.ndarray]:
    """Key every sparse cluster by its anchor atom and Cartesian offsets."""
    scaled = atoms.get_scaled_positions(wrap=False)
    cell = np.asarray(atoms.cell, dtype=float)
    clusters: dict[tuple, np.ndarray] = {}
    for sites, translations, tensor in zip(
        sparse.sites, sparse.translations, sparse.tensors, strict=True
    ):
        anchor = int(sites[0])
        offsets = []
        for site, translation in zip(sites[1:], translations, strict=True):
            offset = (scaled[int(site)] + np.asarray(translation, dtype=float)) - scaled[anchor]
            offsets.append(tuple(np.round(offset @ cell, 6).tolist()))
        clusters[(anchor, tuple(offsets))] = tensor
    return clusters


@pytest.mark.parametrize("shear", sorted(_SHEARS))
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("material", sorted(_MATERIALS))
def test_orbit_subspaces_are_invariant_under_unimodular_rebasing(material, order, shear):
    factory, cutoff = _MATERIALS[material]
    atoms = factory()
    direct = _space(atoms, order, cutoff)
    rebased = _space(_rebased(atoms, _SHEARS[shear]), order, cutoff)

    assert len(rebased.orbits) == len(direct.orbits)
    assert [orbit.dimension for orbit in rebased.orbits] == [
        orbit.dimension for orbit in direct.orbits
    ]
    for left, right in zip(direct.orbits, rebased.orbits, strict=True):
        assert left.observation_rows.tolist() == right.observation_rows.tolist()
        assert left.observation_condition == pytest.approx(right.observation_condition, rel=1e-9)
        np.testing.assert_array_equal(left.exact_lattice_basis, right.exact_lattice_basis)
        np.testing.assert_allclose(left.cartesian_basis, right.cartesian_basis, atol=1e-10)
        np.testing.assert_allclose(
            left.coefficient_transform, right.coefficient_transform, atol=1e-10
        )
        assert left.observation_condition <= 100.0


@pytest.mark.parametrize("shear", sorted(_SHEARS))
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("material", sorted(_MATERIALS))
def test_expansion_is_physically_identical_under_unimodular_rebasing(material, order, shear):
    factory, cutoff = _MATERIALS[material]
    atoms = factory()
    rebased_atoms = _rebased(atoms, _SHEARS[shear])
    direct = _space(atoms, order, cutoff)
    rebased = _space(rebased_atoms, order, cutoff)

    parameters = np.random.default_rng(7).normal(size=direct.n_parameters)
    reference = _physical_clusters(atoms, expand_primitive_parameters(direct, parameters))
    shifted = _physical_clusters(rebased_atoms, expand_primitive_parameters(rebased, parameters))

    assert shifted.keys() == reference.keys()
    for key, tensor in reference.items():
        np.testing.assert_allclose(shifted[key], tensor, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("shear", ["large"])
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("material", ["diamond-silicon", "hexagonal-magnesium"])
def test_identifiability_agrees_under_unimodular_rebasing(material, order, shear):
    factory, cutoff = _MATERIALS[material]
    atoms = factory()
    rebased_atoms = _rebased(atoms, _SHEARS[shear])
    supercell = build_supercell(atoms, (2, 2, 2))
    rebased_supercell = build_supercell(rebased_atoms, (2, 2, 2))

    from mlfcs.structure.relation import StructureRelation

    direct = realize_interaction_space(
        _space(atoms, order, cutoff),
        StructureRelation.from_atoms(atoms, supercell).index,
    )
    rebased = realize_interaction_space(
        _space(rebased_atoms, order, cutoff),
        StructureRelation.from_atoms(rebased_atoms, rebased_supercell).index,
    )
    assert len(rebased.orbits) == len(direct.orbits)
    assert [orbit.dimension for orbit in rebased.orbits] == [
        orbit.dimension for orbit in direct.orbits
    ]


@pytest.mark.parametrize("shear", ["large"])
@pytest.mark.parametrize("material", ["diamond-silicon", "hexagonal-magnesium"])
def test_finite_difference_reconstruction_agrees_under_unimodular_rebasing(material, shear):
    factory, cutoff = _MATERIALS[material]
    atoms = factory()
    rebased_atoms = _rebased(atoms, _SHEARS[shear])
    order, displacement = 3, 0.01

    reference = _physical_sparse(atoms, order, cutoff, displacement)
    shifted = _physical_sparse(rebased_atoms, order, cutoff, displacement)

    assert shifted.keys() == reference.keys()
    for key, tensor in reference.items():
        np.testing.assert_allclose(shifted[key], tensor, rtol=1e-9, atol=1e-10)


def _physical_sparse(atoms: Atoms, order: int, cutoff: float, displacement: float):
    calculation = FiniteDifferenceCalculation(
        atoms,
        reference=build_supercell(atoms, (2, 2, 2)),
        order=order,
        cutoff=cutoff,
        displacement=displacement,
    )
    result = calculation.run(LennardJones(), acoustic_sum_rule=False)
    return _physical_clusters(atoms, result.sparse[order])
