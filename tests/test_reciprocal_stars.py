"""Exact grid-preserving subgroup and star decomposition of the reciprocal q grid.

The stage-B contract of the reciprocal-space symmetry reduction plan:

* only operations that keep the supercell lattice ``Z^3 S`` may act on the grid, and the
  integer compatibility test must agree with the label closure exactly;
* labels transform as ``q' = q R^{-1}`` for the row-vector convention
  ``x' = x R^T + t`` used everywhere in this repository, with no tolerance anywhere;
* time reversal is an explicit antiunitary star operation that can be switched off;
* representatives are the smallest full-grid index, members ascend and the smallest
  operation that reaches a member is recorded, so nothing depends on the order spglib
  returned the operations in;
* the stars partition the grid, the weights sum to ``N_q``, and the little group keeps
  every operation that fixes the representative (operation level, not label level).

Every assertion is an integer statement about labels: no ``allclose`` and no floating
point comparison decides a membership, an orbit or a weight.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlfcs.reciprocal.grid import (
    IrreducibleReciprocalGrid,
    ReciprocalGridSymmetry,
    irreducible_reciprocal_grid,
    reciprocal_grid_symmetry,
    reciprocal_quotient_grid,
    rotate_label,
    rotate_labels,
)
from mlfcs.structure.integer_lattice import (
    adjugate_3x3,
    determinant_3x3,
    supercell_lattice_compatible_indices,
)
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

_SYMPREC = 1e-5


def _monoatomic_cubic() -> Atoms:
    return Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)


def _fcc_primitive() -> Atoms:
    return bulk("Al", "fcc", a=4.05)


def _hexagonal() -> Atoms:
    return bulk("Mg", "hcp", a=3.2, c=5.2)


def _rhombohedral() -> Atoms:
    return bulk("As", "rhombohedral", a=4.1, alpha=54.0)


def _monoclinic_tilted() -> Atoms:
    beta = np.deg2rad(102.0)
    cell = np.asarray(
        [[3.4, 0.0, 0.0], [0.0, 4.1, 0.0], [5.2 * np.cos(beta), 0.0, 5.2 * np.sin(beta)]]
    )
    return Atoms("Si2", scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], cell=cell, pbc=True)


def _diamond() -> Atoms:
    return bulk("Si", "diamond", a=5.43)


def _rocksalt() -> Atoms:
    return bulk("NaCl", "rocksalt", a=5.64)


def _tilted_binary() -> Atoms:
    cell = np.asarray([[3.2, 0.0, 0.0], [1.1, 3.0, 0.0], [0.7, 0.9, 4.5]])
    return Atoms(
        "SiGe",
        scaled_positions=[[0.0, 0.0, 0.0], [0.35, 0.45, 0.55]],
        cell=cell,
        pbc=True,
    )


#: Crystal coverage of plan section 9.2: monoatomic cubic, fcc, hexagonal, rhombohedral
#: (tilted cell), multi-atom, non-symmorphic and a triclinic binary without any spatial
#: symmetry beyond the identity.
_CRYSTALS = {
    "monoatomic-cubic": _monoatomic_cubic,
    "fcc-primitive": _fcc_primitive,
    "hexagonal": _hexagonal,
    "rhombohedral": _rhombohedral,
    "monoclinic-tilted": _monoclinic_tilted,
    "diamond-nonsymmorphic": _diamond,
    "rocksalt-multi-atom": _rocksalt,
    "tilted-binary-p1": _tilted_binary,
}

_SUPERCELLS = {
    "anisotropic": np.diag([2, 3, 4]),
    "tetragonal": np.diag([2, 2, 4]),
    "cubic-three": np.diag([3, 3, 3]),
    "non-diagonal": np.asarray([[2, 1, 0], [0, 2, 1], [0, 0, 2]], dtype=np.int64),
    "sheared": np.asarray([[1, 2, 0], [0, 1, 0], [0, 2, 3]], dtype=np.int64),
}

_GRIDS = tuple(itertools.product(_CRYSTALS, _SUPERCELLS))

#: Cases whose compatible rotations are not orthogonal as integer matrices, so the row
#: vector action ``l R^{-1}`` and the transposed reading ``l R^{-T}`` differ on the grid.
_CONVENTION_CASES = (
    ("fcc-primitive", "sheared"),
    ("hexagonal", "non-diagonal"),
    ("diamond-nonsymmorphic", "sheared"),
)


@cache
def _symmetry(name: str) -> PrimitiveSymmetryOperations:
    return PrimitiveSymmetryOperations.from_atoms(_CRYSTALS[name](), symprec=_SYMPREC)


def _label_map(labels: np.ndarray) -> dict[tuple[int, int, int], int]:
    return {tuple(int(value) for value in row): index for index, row in enumerate(labels)}


def _label_set(labels: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(value) for value in row) for row in labels}


def _integer_inverse(rotation: np.ndarray) -> np.ndarray:
    """Return the exact integer inverse of one unimodular rotation."""
    determinant = determinant_3x3(rotation)
    assert abs(determinant) == 1, rotation
    inverse = adjugate_3x3(rotation)
    return inverse if determinant > 0 else -inverse


def _operation_rows(grid_symmetry: ReciprocalGridSymmetry) -> dict[tuple, int]:
    """Return the row of each operation from its integer rotation."""
    return {
        tuple(int(value) for value in rotation.ravel()): row
        for row, rotation in enumerate(grid_symmetry.rotations.astype(np.int64))
    }


def _permuted_operations(
    symmetry: PrimitiveSymmetryOperations, permutation: np.ndarray
) -> PrimitiveSymmetryOperations:
    """Return the same operations listed in another order."""
    return PrimitiveSymmetryOperations(
        rotations=symmetry.rotations[permutation],
        translations=symmetry.translations[permutation],
        cartesian_rotations=symmetry.cartesian_rotations[permutation],
        site_permutations=symmetry.site_permutations[permutation],
        site_shifts=symmetry.site_shifts[permutation],
        symbol=symmetry.symbol,
    )


def _doubled_operations(symmetry: PrimitiveSymmetryOperations) -> PrimitiveSymmetryOperations:
    """Return ``symmetry`` with every operation listed twice under a shifted translation.

    The two copies of one operation share a rotation and therefore induce the same
    permutation on every label, which is exactly the situation in which an
    operation-level little group and a label-level one disagree.
    """
    return PrimitiveSymmetryOperations(
        rotations=np.concatenate([symmetry.rotations, symmetry.rotations]),
        translations=np.concatenate(
            [symmetry.translations, np.mod(symmetry.translations + 0.5, 1.0)]
        ),
        cartesian_rotations=np.concatenate([symmetry.cartesian_rotations] * 2),
        site_permutations=np.concatenate([symmetry.site_permutations] * 2),
        site_shifts=np.concatenate([symmetry.site_shifts] * 2),
        symbol=symmetry.symbol,
    )


def _negative_index(grid) -> np.ndarray:
    """Return the full-grid index of ``-q`` for every full-grid index."""
    index = _label_map(grid.labels)
    return np.asarray([index[grid.negative_label(label)] for label in grid.labels], dtype=np.int64)


def _reference_orbits(grid, rotations: np.ndarray, *, time_reversal: bool) -> list[list[int]]:
    """Return the stars from the definition, without the permutation machinery."""
    index = _label_map(grid.labels)
    reversed_index = [index[grid.negative_label(label)] for label in grid.labels]
    orbits: list[list[int]] = []
    assigned: set[int] = set()
    for start in range(len(grid.labels)):
        if start in assigned:
            continue
        orbit = {start}
        frontier = [start]
        while frontier:
            member = frontier.pop()
            images = [
                index[rotate_label(grid.labels[member], rotation, grid.denominator)]
                for rotation in rotations
            ]
            if time_reversal:
                images.append(reversed_index[member])
            for image in images:
                if image not in orbit:
                    orbit.add(image)
                    frontier.append(image)
        assigned |= orbit
        orbits.append(sorted(orbit))
    return orbits


def _little_rotations(grid_symmetry: ReciprocalGridSymmetry, result) -> list[set[tuple]]:
    """Return every little group as the set of its integer rotations, order free."""
    return [
        {
            tuple(int(value) for value in grid_symmetry.rotations[int(row)].ravel())
            for row in np.searchsorted(grid_symmetry.operation_indices, group)
        }
        for group in result.little_groups
    ]


def _case(crystal: str, supercell: str):
    """Return ``(matrix, symmetry, full grid, grid symmetry)`` for one test case."""
    matrix = _SUPERCELLS[supercell]
    symmetry = _symmetry(crystal)
    grid = reciprocal_quotient_grid(matrix)
    return matrix, symmetry, grid, reciprocal_grid_symmetry(matrix, symmetry)


# --------------------------------------------------------------------------------------
# 9.1 exact algebra: subgroup, action, partition, weights
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("crystal", "supercell"), _GRIDS)
def test_grid_preserving_subgroup_matches_label_closure(crystal: str, supercell: str) -> None:
    """The extracted integer test must be exactly the closure of the label action."""
    matrix, symmetry, grid, grid_symmetry = _case(crystal, supercell)
    present = _label_set(grid.labels)
    selected = {int(value) for value in grid_symmetry.operation_indices}
    assert np.array_equal(
        grid_symmetry.operation_indices,
        supercell_lattice_compatible_indices(matrix, symmetry.rotations),
    )
    for operation, rotation in enumerate(np.asarray(symmetry.rotations)):
        moved = rotate_labels(grid.labels, rotation, grid.denominator)
        rows = _label_set(moved)
        closed = rows <= present and len(rows) == len(moved) == grid.denominator
        assert closed is (operation in selected), (
            f"{crystal}/{supercell}: operation {operation} with rotation "
            f"{np.asarray(rotation).tolist()} has label closure {closed}"
        )
    # every kept operation is verified per label and never dropped by a tolerance
    for row, operation in enumerate(grid_symmetry.operation_indices):
        moved = rotate_labels(grid.labels, grid_symmetry.rotations[row], grid.denominator)
        assert _label_set(moved) <= present
        assert len(_label_set(moved)) == len(moved) == grid.denominator
        assert grid_symmetry.operation_indices[row] == int(operation)
    assert grid_symmetry.label_permutations.dtype == np.int64
    assert grid_symmetry.label_permutations.shape == (len(selected), grid.denominator)


@pytest.mark.parametrize(("crystal", "supercell"), _GRIDS)
def test_label_permutations_follow_the_public_label_action(crystal: str, supercell: str) -> None:
    """The stored permutations must be the exact action of :func:`rotate_labels`."""
    _, _, grid, grid_symmetry = _case(crystal, supercell)
    index = _label_map(grid.labels)
    for row, rotation in enumerate(grid_symmetry.rotations):
        expected = [
            index[tuple(int(value) for value in label)]
            for label in rotate_labels(grid.labels, rotation, grid.denominator)
        ]
        assert np.array_equal(grid_symmetry.label_permutations[row], expected)


@pytest.mark.parametrize(("crystal", "supercell"), _GRIDS)
def test_label_action_is_a_group_action_on_the_exact_grid(crystal: str, supercell: str) -> None:
    """Identity, inverse and composition close on the labels, as index permutations."""
    _, _, grid, grid_symmetry = _case(crystal, supercell)
    permutations = grid_symmetry.label_permutations
    rotations = grid_symmetry.rotations.astype(np.int64)
    rows = _operation_rows(grid_symmetry)
    count = len(grid.labels)
    identity = rows[tuple(int(value) for value in np.eye(3, dtype=np.int64).ravel())]
    assert np.array_equal(permutations[identity], np.arange(count, dtype=np.int64))
    for first in range(len(rotations)):
        inverse = _integer_inverse(rotations[first])
        inverse_row = rows[tuple(int(value) for value in inverse.ravel())]
        assert np.array_equal(
            permutations[inverse_row][permutations[first]], np.arange(count, dtype=np.int64)
        )
        for second in range(len(rotations)):
            product = rows[
                tuple(int(value) for value in (rotations[first] @ rotations[second]).ravel())
            ]
            assert np.array_equal(permutations[product], permutations[first][permutations[second]])


@pytest.mark.parametrize(("crystal", "supercell"), _GRIDS)
def test_stars_partition_the_grid_with_exact_weights(crystal: str, supercell: str) -> None:
    """Stars are a disjoint partition, weights are the star sizes and sum to N_q."""
    matrix, symmetry, grid, _ = _case(crystal, supercell)
    result = irreducible_reciprocal_grid(matrix, symmetry)
    count = grid.denominator
    assert np.array_equal(result.full.labels, grid.labels)
    assert result.full.denominator == count == len(grid.labels)
    members = np.concatenate([star.members for star in result.stars])
    assert np.array_equal(np.sort(members), np.arange(count, dtype=np.int64))
    assert len(members) == count
    assert np.all(result.weights == np.asarray([len(star.members) for star in result.stars]))
    assert int(result.weights.sum()) == count
    assert np.all(np.diff(result.representatives) > 0)
    for index, star in enumerate(result.stars):
        assert star.members.dtype == np.int64
        assert star.operations.dtype == np.int64
        assert star.antiunitary.dtype == np.bool_
        assert int(star.members[0]) == star.representative == int(result.representatives[index])
        assert np.all(np.diff(star.members) > 0)
        assert np.all(result.full_to_irreducible[star.members] == index)
        assert np.array_equal(result.full_operations[star.members], star.operations)
        assert np.array_equal(result.full_antiunitary[star.members], star.antiunitary)


@pytest.mark.parametrize(("crystal", "supercell"), _GRIDS)
@pytest.mark.parametrize("time_reversal", [True, False], ids=["reversal", "no-reversal"])
def test_stars_are_the_orbits_of_the_definition(
    crystal: str, supercell: str, time_reversal: bool
) -> None:
    """An independent orbit walk from the label definition reproduces every star."""
    matrix, symmetry, grid, grid_symmetry = _case(crystal, supercell)
    result = irreducible_reciprocal_grid(matrix, symmetry, time_reversal=time_reversal)
    orbits = _reference_orbits(grid, grid_symmetry.rotations, time_reversal=time_reversal)
    assert [star.members.tolist() for star in result.stars] == orbits
    assert [star.representative for star in result.stars] == [members[0] for members in orbits]
    if not time_reversal:
        assert not np.any(result.full_antiunitary)


# --------------------------------------------------------------------------------------
# 5.2 the exact label action and its row vector convention
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("crystal", "supercell"), _GRIDS)
def test_phase_invariance_pins_the_row_vector_label_action(crystal: str, supercell: str) -> None:
    """``q' . (T R^T) = q . T`` holds exactly on every supercell lattice vector T."""
    matrix, _, grid, grid_symmetry = _case(crystal, supercell)
    lattice = np.asarray(
        [[n0, n1, n2] @ matrix for n0, n1, n2 in itertools.product((-1, 0, 1), repeat=3)],
        dtype=np.int64,
    )
    for row in range(len(grid_symmetry.operation_indices)):
        rotation = grid_symmetry.rotations[row].astype(np.int64)
        moved = rotate_labels(grid.labels, rotation, grid.denominator)
        # the reduced label differs from l R^{-1} by a multiple of D, so the Bloch phase
        # statement q' . (T R^T) = q . T is exact modulo the denominator
        assert np.array_equal(
            np.mod(moved @ (lattice @ rotation.T).T, grid.denominator),
            np.mod(grid.labels @ lattice.T, grid.denominator),
        )
        assert np.array_equal(
            np.mod(moved @ rotation, grid.denominator),
            np.mod(grid.labels, grid.denominator),
        )


@pytest.mark.parametrize(("crystal", "supercell"), _CONVENTION_CASES)
def test_transposed_label_action_is_rejected(crystal: str, supercell: str) -> None:
    """A non-diagonal supercell separates the two conventions and the transpose fails."""
    _, _, grid, grid_symmetry = _case(crystal, supercell)
    differing = 0
    violating = 0
    for row in range(len(grid_symmetry.operation_indices)):
        rotation = grid_symmetry.rotations[row].astype(np.int64)
        moved = rotate_labels(grid.labels, rotation, grid.denominator)
        transposed = np.mod(grid.labels @ rotation.T, grid.denominator)
        if not np.array_equal(transposed, moved):
            differing += 1
        if not np.array_equal(
            np.mod(transposed @ rotation, grid.denominator),
            np.mod(grid.labels, grid.denominator),
        ):
            violating += 1
    assert differing >= 1
    assert violating >= 1, "the transposed reading must break q' R = q for this case"


@pytest.mark.parametrize("crystal", ["monoatomic-cubic", "fcc-primitive", "tilted-binary-p1"])
def test_rotate_labels_is_exact_and_shape_preserving(crystal: str) -> None:
    """The public action keeps shapes, agrees with its scalar form and rejects floats."""
    symmetry = _symmetry(crystal)
    grid = reciprocal_quotient_grid(_SUPERCELLS["non-diagonal"])
    identity = np.eye(3, dtype=np.int64)
    # P1 has only the identity, so a unimodular shear stands in for the rotation there
    rotation = next(
        (
            np.asarray(candidate, dtype=np.int64)
            for candidate in symmetry.rotations
            if not np.array_equal(candidate, identity)
        ),
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64),
    )
    assert not np.array_equal(rotation, identity)
    moved = rotate_labels(grid.labels, rotation, grid.denominator)
    assert moved.shape == grid.labels.shape
    assert moved.dtype == np.int64
    for label, image in zip(grid.labels, moved, strict=True):
        assert rotate_label(label, rotation, grid.denominator) == (
            int(image[0]),
            int(image[1]),
            int(image[2]),
        )
    block = np.stack([grid.labels, grid.labels])
    assert rotate_labels(block, rotation, grid.denominator).shape == block.shape
    assert np.array_equal(
        rotate_labels(block, rotation, grid.denominator)[1],
        rotate_labels(grid.labels, rotation, grid.denominator),
    )
    with pytest.raises(ValueError, match="integer array"):
        rotate_labels(grid.labels.astype(float), rotation, grid.denominator)
    with pytest.raises(ValueError, match="integer array"):
        rotate_labels(grid.labels, rotation.astype(float), grid.denominator)
    with pytest.raises(ValueError, match="unimodular"):
        rotate_labels(grid.labels, np.diag([2, 1, 1]), grid.denominator)
    with pytest.raises(ValueError, match="shape"):
        rotate_labels(grid.labels[:, :2], rotation, grid.denominator)
    with pytest.raises(ValueError, match="positive integer"):
        rotate_labels(grid.labels, rotation, 0)
    with pytest.raises(ValueError, match="shape"):
        rotate_label(grid.labels[0][:2], rotation, grid.denominator)


def test_time_reversal_uses_the_exact_negative_label() -> None:
    """Time reversal is the existing negative label, never ``-x`` on floats."""
    grid = reciprocal_quotient_grid(_SUPERCELLS["non-diagonal"])
    negated = np.mod(-grid.labels, grid.denominator)
    for label, image in zip(grid.labels, negated, strict=True):
        assert grid.negative_label(label) == tuple(int(value) for value in image)
    assert np.array_equal(np.mod(-negated, grid.denominator), grid.labels)


# --------------------------------------------------------------------------------------
# 9.1 anisotropic and multiplied grids
# --------------------------------------------------------------------------------------


def test_anisotropic_supercell_keeps_only_the_compatible_subgroup() -> None:
    """Axis densities that differ forbid the axis permutations of the cubic group."""
    matrix = _SUPERCELLS["anisotropic"]
    symmetry = _symmetry("monoatomic-cubic")
    grid_symmetry = reciprocal_grid_symmetry(matrix, symmetry)
    assert symmetry.size == 48
    assert grid_symmetry.operation_indices.size == 8
    kept = {tuple(int(value) for value in rotation.ravel()) for rotation in grid_symmetry.rotations}
    diagonal_signs = {
        tuple(int(value) for value in np.diag(signs).ravel())
        for signs in itertools.product((1, -1), repeat=3)
    }
    assert kept == diagonal_signs
    full = {tuple(int(value) for value in rotation.ravel()) for rotation in symmetry.rotations}
    for quarter_turn in (
        np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.int64),
        np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.int64),
    ):
        key = tuple(int(value) for value in quarter_turn.ravel())
        assert key in full and key not in kept
    assert len(reciprocal_quotient_grid(matrix).labels) == 24
    result = irreducible_reciprocal_grid(matrix, symmetry)
    assert int(result.weights.sum()) == 24
    assert not np.any(result.full_antiunitary)


@pytest.mark.parametrize("crystal", sorted(_CRYSTALS))
@pytest.mark.parametrize("multiplier", [2, 3], ids=["x2", "x3"])
def test_multiplied_grid_keeps_the_compatibility_decision(crystal: str, multiplier: int) -> None:
    """A uniformly refined mesh keeps the subgroup and refines every star exactly."""
    matrix = _SUPERCELLS["non-diagonal"]
    symmetry = _symmetry(crystal)
    grid_symmetry = reciprocal_grid_symmetry(matrix, symmetry)
    coarse = irreducible_reciprocal_grid(matrix, symmetry)
    fine = irreducible_reciprocal_grid(matrix * multiplier, symmetry)
    embedding = multiplier**3
    assert np.array_equal(
        supercell_lattice_compatible_indices(matrix, symmetry.rotations),
        supercell_lattice_compatible_indices(matrix * multiplier, symmetry.rotations),
    )
    assert len(fine.full.labels) == embedding * len(coarse.full.labels)
    fine_index = _label_map(fine.full.labels)
    for label in coarse.full.labels:
        for row in range(len(grid_symmetry.operation_indices)):
            rotation = grid_symmetry.rotations[row]
            assert np.array_equal(
                rotate_labels(embedding * label, rotation, fine.full.denominator),
                embedding * rotate_labels(label, rotation, coarse.full.denominator),
            )
    for star in coarse.stars:
        representative = tuple(
            embedding * int(value) for value in coarse.full.labels[star.representative]
        )
        refined = fine.stars[int(fine.full_to_irreducible[fine_index[representative]])]
        assert _label_set(fine.full.labels[refined.members]) == {
            tuple(embedding * int(value) for value in coarse.full.labels[member])
            for member in star.members
        }


# --------------------------------------------------------------------------------------
# 5.3 time reversal
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("crystal", "supercell"), _GRIDS)
def test_time_reversal_flags_are_recorded_and_switchable(crystal: str, supercell: str) -> None:
    """Every star is time reversal closed and the flag marks the antiunitary maps."""
    matrix, symmetry, grid, grid_symmetry = _case(crystal, supercell)
    reversed_index = np.asarray(
        [_label_map(grid.labels)[grid.negative_label(label)] for label in grid.labels],
        dtype=np.int64,
    )
    with_reversal = irreducible_reciprocal_grid(matrix, symmetry, time_reversal=True)
    without_reversal = irreducible_reciprocal_grid(matrix, symmetry, time_reversal=False)
    assert not np.any(without_reversal.full_antiunitary)
    assert len(without_reversal.representatives) >= len(with_reversal.representatives)
    for star in with_reversal.stars:
        assert set(reversed_index[star.members].tolist()) == set(star.members.tolist())
        unitarily_reached = grid_symmetry.label_permutations[:, star.representative]
        for position, member in enumerate(star.members):
            mapped = np.flatnonzero(unitarily_reached == member)
            assert bool(star.antiunitary[position]) is (len(mapped) == 0)
            if len(mapped):
                assert int(star.operations[position]) == int(
                    grid_symmetry.operation_indices[mapped[0]]
                )
            else:
                row = int(
                    np.searchsorted(grid_symmetry.operation_indices, int(star.operations[position]))
                )
                assert (
                    grid_symmetry.label_permutations[row, int(reversed_index[star.representative])]
                    == member
                )
    for star in without_reversal.stars:
        owner = int(with_reversal.full_to_irreducible[star.representative])
        assert set(star.members.tolist()) <= set(with_reversal.stars[owner].members.tolist())


def test_time_reversal_creates_the_q_and_minus_q_pairs_of_a_nonsymmetric_crystal() -> None:
    """Without a spatial group only time reversal pairs q with -q, and it can be off."""
    matrix = _SUPERCELLS["anisotropic"]
    symmetry = _symmetry("tilted-binary-p1")
    assert symmetry.size == 1
    grid = reciprocal_quotient_grid(matrix)
    paired = irreducible_reciprocal_grid(matrix, symmetry, time_reversal=True)
    unpaired = irreducible_reciprocal_grid(matrix, symmetry, time_reversal=False)
    assert len(unpaired.representatives) == len(grid.labels) == 24
    assert len(paired.representatives) == 14
    assert int(paired.weights.sum()) == 24
    sizes = sorted(int(value) for value in paired.weights)
    assert sizes.count(2) == 10 and sizes.count(1) == 4
    assert int(paired.full_antiunitary.sum()) == 10
    for star in paired.stars:
        if len(star.members) == 2:
            assert _label_set(np.mod(-grid.labels[star.members], grid.denominator)) == _label_set(
                grid.labels[star.members]
            )
            assert star.antiunitary.tolist() == [False, True]
            assert star.operations.tolist() == [0, 0]
        else:
            assert star.members.size == 1
            assert grid.negative_label(grid.labels[star.representative]) == tuple(
                int(value) for value in grid.labels[star.representative]
            )


# --------------------------------------------------------------------------------------
# 9.3 special q points and little groups
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("crystal", "supercell"), _GRIDS)
def test_gamma_and_boundary_points_keep_their_little_groups(crystal: str, supercell: str) -> None:
    """Gamma, ``q = -q + G`` points and every star carry the right stabilizer."""
    matrix, symmetry, grid, grid_symmetry = _case(crystal, supercell)
    result = irreducible_reciprocal_grid(matrix, symmetry)
    gamma = int(result.full_to_irreducible[0])
    assert int(result.representatives[gamma]) == 0
    assert result.stars[gamma].members.tolist() == [0]
    assert np.array_equal(result.little_groups[gamma], grid_symmetry.operation_indices)
    assert int(result.weights[gamma]) == 1
    boundary = [
        index
        for index, label in enumerate(grid.labels)
        if grid.negative_label(label) == tuple(int(value) for value in label)
    ]
    assert boundary, "an even grid always contains q = -q + G"
    for index in boundary:
        star = result.stars[int(result.full_to_irreducible[index])]
        assert _label_set(np.mod(-grid.labels[star.members], grid.denominator)) == _label_set(
            grid.labels[star.members]
        )
        assert not np.any(result.full_antiunitary[star.members])
    for index, star in enumerate(result.stars):
        fixing = grid_symmetry.label_permutations[:, star.representative] == star.representative
        assert np.array_equal(result.little_groups[index], grid_symmetry.operation_indices[fixing])
        assert len(result.little_groups[index]) >= 1


def test_cubic_three_mesh_has_the_classic_high_symmetry_star_sizes() -> None:
    """A 3x3x3 cubic mesh gives one Gamma point, 6 faces, 8 corners and 12 edges."""
    matrix = np.diag([3, 3, 3])
    symmetry = _symmetry("monoatomic-cubic")
    result = irreducible_reciprocal_grid(matrix, symmetry)
    assert len(result.full.labels) == 27
    assert len(result.representatives) == 4
    assert sorted(int(value) for value in result.weights) == [1, 6, 8, 12]
    assert sorted(len(group) for group in result.little_groups) == [4, 6, 8, 48]
    assert int(result.weights.sum()) == 27
    center = result.stars[int(result.full_to_irreducible[0])]
    assert center.members.tolist() == [0]
    assert center.operations.tolist() == [0]
    assert center.antiunitary.tolist() == [False]
    face_label = np.asarray([0, 0, 9], dtype=np.int64)
    face_index = int(np.flatnonzero(np.all(result.full.labels == face_label, axis=1))[0])
    faces = result.stars[int(result.full_to_irreducible[face_index])]
    assert len(faces.members) == 6
    assert not np.any(faces.antiunitary)


# --------------------------------------------------------------------------------------
# canonical construction: independent of the operation order
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("crystal", "supercell"), _GRIDS)
def test_permuted_operation_order_gives_the_same_decomposition(
    crystal: str, supercell: str
) -> None:
    """Spglib's accidental order must not reach the stars, weights or little groups.

    The *index* recorded per member is by construction an index into the given operation
    list, so it follows a relabelling; everything a consumer reads from the
    decomposition -- stars, members, weights, the reverse map, the antiunitary flags and
    the little-group rotations -- is invariant, and a member a single operation reaches
    keeps the same rotation.
    """
    matrix = _SUPERCELLS[supercell]
    symmetry = _symmetry(crystal)
    permutation = np.random.default_rng(20260921).permutation(symmetry.size)
    shuffled = _permuted_operations(symmetry, permutation)
    matrix_grid = reciprocal_quotient_grid(matrix)
    negated = _negative_index(matrix_grid)
    reference = irreducible_reciprocal_grid(matrix, symmetry)
    candidate = irreducible_reciprocal_grid(matrix, shuffled)
    reference_symmetry = reciprocal_grid_symmetry(matrix, symmetry)
    candidate_symmetry = reciprocal_grid_symmetry(matrix, shuffled)
    assert [star.members.tolist() for star in candidate.stars] == [
        star.members.tolist() for star in reference.stars
    ]
    assert np.array_equal(candidate.representatives, reference.representatives)
    assert np.array_equal(candidate.weights, reference.weights)
    assert np.array_equal(candidate.full_to_irreducible, reference.full_to_irreducible)
    assert np.array_equal(candidate.full_antiunitary, reference.full_antiunitary)
    assert {tuple(row) for row in candidate_symmetry.label_permutations} == {
        tuple(row) for row in reference_symmetry.label_permutations
    }
    assert _little_rotations(reference_symmetry, reference) == _little_rotations(
        candidate_symmetry, candidate
    )
    for grid_symmetry, result in (
        (reference_symmetry, reference),
        (candidate_symmetry, candidate),
    ):
        for star in result.stars:
            unitary = grid_symmetry.label_permutations[:, star.representative]
            reversed_unitary = grid_symmetry.label_permutations[
                :, int(negated[star.representative])
            ]
            for position, member in enumerate(star.members):
                mapped = np.flatnonzero(unitary == member)
                assert bool(star.antiunitary[position]) is (len(mapped) == 0)
                rows = mapped if len(mapped) else np.flatnonzero(reversed_unitary == member)
                assert int(star.operations[position]) == int(
                    grid_symmetry.operation_indices[rows[0]]
                )
    for star in reference.stars:
        for position, member in enumerate(star.members):
            if star.antiunitary[position]:
                continue
            mapped = np.flatnonzero(
                reference_symmetry.label_permutations[:, star.representative] == member
            )
            if len(mapped) != 1:
                continue
            other = np.flatnonzero(
                candidate_symmetry.label_permutations[:, star.representative] == member
            )
            assert len(other) == 1
            assert np.array_equal(
                np.asarray(candidate_symmetry.rotations[other[0]]),
                np.asarray(reference_symmetry.rotations[mapped[0]]),
            )
    assert len(reference.representatives) == len(candidate.representatives)


# --------------------------------------------------------------------------------------
# 6 operation-level little group, distinguished from the label level
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("crystal", "supercell"), _GRIDS)
def test_little_group_is_exactly_the_operation_level_stabilizer(
    crystal: str, supercell: str
) -> None:
    """The little group is every operation that fixes the representative label."""
    matrix, symmetry, grid, grid_symmetry = _case(crystal, supercell)
    result = irreducible_reciprocal_grid(matrix, symmetry, time_reversal=False)
    rotations = grid_symmetry.rotations.astype(np.int64)
    rows = _operation_rows(grid_symmetry)
    identity = int(
        grid_symmetry.operation_indices[
            rows[tuple(int(value) for value in np.eye(3, dtype=np.int64).ravel())]
        ]
    )
    for index, star in enumerate(result.stars):
        fixing = grid_symmetry.label_permutations[:, star.representative] == star.representative
        assert np.array_equal(result.little_groups[index], grid_symmetry.operation_indices[fixing])
        little = {int(operation) for operation in result.little_groups[index]}
        assert identity in little
        for first, second in itertools.product(sorted(little), repeat=2):
            first_row = int(np.searchsorted(grid_symmetry.operation_indices, first))
            second_row = int(np.searchsorted(grid_symmetry.operation_indices, second))
            product = rotations[first_row] @ rotations[second_row]
            product_row = rows[tuple(int(value) for value in product.ravel())]
            assert int(grid_symmetry.operation_indices[product_row]) in little
    assert len(grid.labels) == grid.denominator


def test_little_group_keeps_operations_that_coincide_on_every_label() -> None:
    """Two operations with one rotation are both kept in the operation-level group."""
    matrix = _SUPERCELLS["non-diagonal"]
    symmetry = _symmetry("fcc-primitive")
    doubled = _doubled_operations(symmetry)
    single = irreducible_reciprocal_grid(matrix, symmetry, time_reversal=False)
    paired = irreducible_reciprocal_grid(matrix, doubled, time_reversal=False)
    assert doubled.size == 2 * symmetry.size
    grid_symmetry = reciprocal_grid_symmetry(matrix, doubled)
    compatible = len(grid_symmetry.operation_indices) // 2
    assert np.array_equal(
        grid_symmetry.label_permutations[:compatible],
        grid_symmetry.label_permutations[compatible:],
    )
    assert np.array_equal(
        grid_symmetry.operation_indices[compatible:] - symmetry.size,
        grid_symmetry.operation_indices[:compatible],
    )
    assert [star.members.tolist() for star in paired.stars] == [
        star.members.tolist() for star in single.stars
    ]
    for index, star in enumerate(single.stars):
        operations = paired.little_groups[index]
        assert len(operations) == 2 * len(single.little_groups[index])
        fixing = grid_symmetry.label_permutations[:, star.representative] == star.representative
        assert np.array_equal(operations, grid_symmetry.operation_indices[fixing])
    distinct = {tuple(permutation) for permutation in grid_symmetry.label_permutations}
    assert len(distinct) < len(grid_symmetry.label_permutations)


# --------------------------------------------------------------------------------------
# 11 locating failures instead of silent wrong results
# --------------------------------------------------------------------------------------


def test_decomposition_rejects_broken_invariants_with_locators() -> None:
    """Corrupting a result must raise with the offending labels, not pass quietly."""
    matrix = _SUPERCELLS["non-diagonal"]
    symmetry = _symmetry("fcc-primitive")
    result = irreducible_reciprocal_grid(matrix, symmetry)
    star = result.stars[-1]
    shortened = replace(
        star,
        members=star.members[:-1],
        operations=star.operations[:-1],
        antiunitary=star.antiunitary[:-1],
    )
    with pytest.raises(RuntimeError, match="partition"):
        replace(result, stars=(*result.stars[:-1], shortened))
    with pytest.raises(RuntimeError, match="weights"):
        replace(result, weights=result.weights + 1)
    with pytest.raises(RuntimeError, match="representatives"):
        replace(result, representatives=result.representatives[::-1])
    with pytest.raises(RuntimeError, match="little group"):
        replace(result, little_groups=(np.zeros(0, dtype=np.int64), *result.little_groups[1:]))
    with pytest.raises(RuntimeError, match="does not cover"):
        replace(result, full_operations=result.full_operations[:-1])
    with pytest.raises(RuntimeError, match="must ascend"):
        replace(star, members=star.members[::-1], operations=star.operations[::-1])
    grid_symmetry = reciprocal_grid_symmetry(matrix, symmetry)
    duplicated = grid_symmetry.label_permutations.copy()
    duplicated[0, 0] = duplicated[0, 1]
    with pytest.raises(RuntimeError, match="does not permute"):
        replace(grid_symmetry, label_permutations=duplicated)
    with pytest.raises(RuntimeError, match="ascend without repeats"):
        replace(grid_symmetry, operation_indices=grid_symmetry.operation_indices[::-1])
    with pytest.raises(RuntimeError, match="rotations"):
        replace(grid_symmetry, rotations=grid_symmetry.rotations[:1])
    assert isinstance(result, IrreducibleReciprocalGrid)
    assert isinstance(grid_symmetry, ReciprocalGridSymmetry)
