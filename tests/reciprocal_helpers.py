"""Shared crystal fixtures and symmetry checks for the reciprocal-space tests.

The fixtures are finite-difference FC2s of a Lennard-Jones crystal.  A finite-difference
reconstruction is symmetry-expanded by construction, so its dynamical matrix is exactly
covariant under the space group; the covalent part of the LJ model is what makes a
Cartesian-rotation error in a representation observable, which a nearest-neighbour spring
network with isotropic tensors would hide.

Every fixture is checked with :func:`symmetry_residual` before it is returned.  The check
uses the library's own supercell atom permutations, so it is independent of the translation
arithmetic written for the reciprocal tests, and a fixture that is not symmetric is refused
rather than silently used to weaken a symmetry test.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.neighborlist import neighbor_list

from mlfcs import FiniteDifferenceCalculation
from mlfcs.tools.supercell import build_supercell
from mlfcs.force_constants.representation import ForceConstants
from mlfcs.structure.relation import StructureRelation
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations, SymmetryOperations

SUPERCELLS = {
    "diagonal": np.diag((2, 2, 2)).astype(np.int64),
    "non-diagonal": np.asarray([[2, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64),
    "anisotropic": np.diag((2, 1, 3)).astype(np.int64),
}

#: Crystals with a non-trivial point group, including a skewed cell whose Cartesian
#: rotations are irrational while its integer rotations are not orthogonal.
CASES = ("cubic", "diamond", "hexagonal", "skewed")

LENNARD_JONES = LennardJones(epsilon=1.0, sigma=3.0, rc=9.0)


def rebased(atoms: Atoms, transform: np.ndarray) -> Atoms:
    """Return the same crystal expressed in the basis ``transform @ cell``."""
    cell = np.asarray(atoms.cell)
    scaled = np.mod(atoms.get_scaled_positions(wrap=True) @ np.linalg.inv(transform), 1.0)
    return Atoms(atoms.numbers, scaled_positions=scaled, cell=transform @ cell, pbc=True)


def primitive(name: str) -> Atoms:
    """Return one test crystal's primitive cell."""
    if name == "cubic":
        return bulk("Ar", "fcc", a=5.26)
    if name == "diamond":
        return bulk("Si", "diamond", a=5.43)
    if name == "hexagonal":
        return bulk("Mg", "hcp", a=3.2, c=5.2)
    if name == "skewed":
        shear = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
        return rebased(bulk("Mg", "hcp", a=3.2, c=5.2), shear)
    raise AssertionError(name)


def symmetry(atoms: Atoms, *, symprec: float = 1e-5) -> PrimitiveSymmetryOperations:
    """Return the space-group operations of ``atoms``."""
    return PrimitiveSymmetryOperations.from_atoms(atoms, symprec=symprec)


def symmetry_residual(
    force_constants: ForceConstants, operations: PrimitiveSymmetryOperations
) -> float:
    """Return the largest violation of the space-group relation over the whole supercell.

    A dense block ``dense[site, atom]`` couples the representative atom of ``site`` to
    ``atom``.  Its image couples the image of that representative to the image of ``atom``,
    and the image of the representative is in general *another* atom of the image site, so
    the pair has to be translated back onto that site's representative before it can index
    the dense array.  Dropping that translation makes every non-diagonal crystal look
    asymmetric, which is why the check is validated against an isotropic control model in
    the tests instead of being trusted on inspection.
    """
    relation = force_constants.relation
    dense = force_constants.materialize(2, max_bytes=None)
    index = relation.index
    permutations = SymmetryOperations.from_primitive_operations(operations, index)
    representatives = np.asarray(
        [index.representative(site) for site in range(len(relation.primitive))], dtype=np.int64
    )
    sites = np.asarray(index.primitive, dtype=np.int64)
    worst = 0.0
    for operation in range(permutations.size):
        permutation = np.asarray(permutations.atom_permutations[operation], dtype=np.int64)
        rotation = np.asarray(permutations.cartesian_rotations[operation], dtype=float)
        image_representatives = permutation[representatives]
        image_sites = sites[image_representatives]
        for site in range(len(relation.primitive)):
            row = int(image_representatives[site])
            for atom in range(len(relation.reference)):
                target = index.anchor((row, int(permutation[atom])))[1]
                block = rotation.T @ dense[site, atom] @ rotation
                worst = max(
                    worst, float(np.max(np.abs(block - dense[int(image_sites[site]), target])))
                )
    return worst


def finite_difference_force_constants(
    primitive_cell: Atoms, matrix: np.ndarray, *, displacement: float = 0.02
) -> ForceConstants:
    """Return the symmetry-expanded FC2 of a Lennard-Jones crystal in a supercell.

    The result is verified to satisfy the space-group relation, so a caller can attribute a
    covariance residual to the representation under test instead of to a broken fixture.
    """
    reference = build_supercell(primitive_cell, matrix)
    calculation = FiniteDifferenceCalculation(
        primitive_cell, reference=reference, order=2, cutoff=-1, displacement=displacement
    )
    force_constants = calculation.run(LENNARD_JONES, acoustic_sum_rule=False)
    residual = symmetry_residual(force_constants, symmetry(force_constants.relation.primitive))
    if residual > 1e-8:
        raise RuntimeError(
            f"the fixture is not crystal symmetric (residual {residual:.3e}); refusing to "
            "use it as a covariance fixture"
        )
    return force_constants


@cache
def _cached_case(
    name: str, supercell_key: str
) -> tuple[str, ForceConstants, Atoms, PrimitiveSymmetryOperations]:
    cell = primitive(name)
    force_constants = finite_difference_force_constants(cell, SUPERCELLS[supercell_key])
    return name, force_constants, cell, symmetry(cell)


def crystal_case(
    name: str, supercell: str | np.ndarray | None = None
) -> tuple[str, ForceConstants, Atoms, PrimitiveSymmetryOperations]:
    """Return ``(name, fc2, primitive, symmetry)`` for one crystal.

    The finite-difference fixture costs a calculator run, so cases named by the ``SUPERCELLS``
    keys are cached and the *same* objects are handed to every caller: treat the returned
    structures and force constants as read-only.  A one-off matrix is built uncached.
    """
    if supercell is None:
        return _cached_case(name, "diagonal")
    if isinstance(supercell, str):
        return _cached_case(name, supercell)
    matrix = np.asarray(supercell)
    cell = primitive(name)
    return name, finite_difference_force_constants(cell, matrix), cell, symmetry(cell)


def relation_reference(cell: Atoms, supercell: np.ndarray) -> Atoms:
    """Return the reference supercell of a structure relation, with its periodic metadata.

    The relation's reference carries the per-atom ``cell_translation`` and
    ``primitive_index`` arrays and the supercell-matrix metadata that the samplers read, so
    the bare builder output is not a drop-in substitute.
    """
    return StructureRelation.from_atoms(
        cell, build_supercell(cell, np.asarray(supercell)), symprec=1e-5
    ).reference


def degenerate_clusters(values: np.ndarray, tolerance: float = 1e-8) -> list[np.ndarray]:
    """Return index arrays of eigenvalues that are equal within ``tolerance``."""
    clusters: list[np.ndarray] = []
    for index, value in enumerate(values):
        if clusters and abs(value - values[clusters[-1][0]]) <= tolerance:
            clusters[-1] = np.append(clusters[-1], index)
        else:
            clusters.append(np.asarray([index]))
    return clusters


def sorted_pairs(values: np.ndarray) -> np.ndarray:
    """Return a sorted copy, for comparing multisets without a canonical order."""
    return np.sort(np.asarray(values, dtype=float))


def assert_same_multiset(left: np.ndarray, right: np.ndarray, *, tolerance: float) -> None:
    """Assert two multisets agree, without relying on a canonical ordering."""
    first, second = sorted_pairs(left), sorted_pairs(right)
    assert first.shape == second.shape
    np.testing.assert_allclose(first, second, rtol=0.0, atol=tolerance)


def shell_bonds(atoms: Atoms, cutoff: float) -> tuple[tuple[int, int, tuple[int, int, int]], ...]:
    """Return every unordered bond inside ``cutoff`` once.

    The bond vector is ``(scaled_b + shift - scaled_a) @ cell`` in Cartesian angstrom; the
    fractional positions and the integer shift are never mixed.
    """
    first, second, shifts, distances = neighbor_list("ijSd", atoms, cutoff, self_interaction=True)
    bonds = set()
    for atom_a, atom_b, shift, distance in zip(
        first.tolist(), second.tolist(), shifts.tolist(), distances.tolist(), strict=True
    ):
        if float(distance) >= cutoff:
            continue
        forward = (int(atom_a), int(atom_b), tuple(int(value) for value in shift))
        backward = (int(atom_b), int(atom_a), tuple(-value for value in forward[2]))
        bonds.add(min(forward, backward))
    return tuple(sorted(bonds))


__all__ = [
    "CASES",
    "LENNARD_JONES",
    "SUPERCELLS",
    "assert_same_multiset",
    "crystal_case",
    "degenerate_clusters",
    "finite_difference_force_constants",
    "primitive",
    "rebased",
    "relation_reference",
    "shell_bonds",
    "sorted_pairs",
    "symmetry",
    "symmetry_residual",
]
