import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlfcs import FiniteDifferenceCalculation, build_supercell, realize_force_constants
from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants
from mlfcs.interactions.algebra.actions import scaled_to_cartesian_matrix
from mlfcs.interactions.algebra.exact import RANK_PRIMES as _RANK_PRIMES
from mlfcs.interactions.algebra.exact import certified_rank
from mlfcs.interactions.keys import InteractionKey
from mlfcs.interactions.primitive.builder import (
    build_primitive_interaction_space,
    primitive_generators,
)
from mlfcs.interactions.realization import (
    InteractionAliasingError,
    realize_interaction_space,
    validate_realization_identifiability,
)
from mlfcs.structure.integer_lattice import same_residue
from mlfcs.structure.relation import StructureRelation
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations


def _observed_block_is_independent(frame: np.ndarray, orbit) -> bool:
    """Return whether the observed component rows determine the orbit parameters."""
    block = (frame @ np.asarray(orbit.basis, dtype=np.int64))[orbit.pivots]
    return bool(np.linalg.matrix_rank(block, tol=1e-9 * np.max(np.abs(block))) == orbit.dimension)


def test_primitive_fc2_space_keeps_exact_nearest_neighbor_translations():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    space = build_primitive_interaction_space(
        primitive,
        order=2,
        cutoff=4.1,
        max_body_order=None,
        symprec=1e-5,
    )

    images = {image.key for orbit in space.orbits for image in orbit.images}
    expected = {
        InteractionKey((0, 0), ((0, 0, 0),)),
        InteractionKey((0, 0), ((1, 0, 0),)),
        InteractionKey((0, 0), ((-1, 0, 0),)),
        InteractionKey((0, 0), ((0, 1, 0),)),
        InteractionKey((0, 0), ((0, -1, 0),)),
        InteractionKey((0, 0), ((0, 0, 1),)),
        InteractionKey((0, 0), ((0, 0, -1),)),
    }
    assert images == expected


@pytest.mark.parametrize("order", [2, 3, 4])
def test_primitive_orbit_bases_are_invariant_and_observed_components_determine(order):
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    space = build_primitive_interaction_space(
        primitive,
        order=order,
        cutoff=4.1,
        max_body_order=2,
        symprec=1e-5,
    )
    frame = scaled_to_cartesian_matrix(space.cell, order)

    for orbit in space.orbits:
        assert _observed_block_is_independent(frame, orbit)
        for image in orbit.images:
            if image.key == orbit.representative:
                np.testing.assert_array_equal(
                    image.action.apply_scaled_columns(orbit.basis),
                    orbit.basis,
                )
                np.testing.assert_allclose(
                    image.action.apply_columns(frame @ orbit.basis),
                    frame @ orbit.basis,
                    rtol=1e-9,
                    atol=1e-9,
                )


def test_exact_ifcs_realize_into_a_different_supercell_size():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    source = build_supercell(primitive, (3, 3, 3))
    calculation = FiniteDifferenceCalculation(
        primitive,
        reference=source,
        order=2,
        cutoff=4.1,
    )
    result = calculation.reap(
        np.zeros((len(calculation.plan), len(source), 3)), acoustic_sum_rule=False
    )
    target = build_supercell(primitive, (2, 2, 2))
    realized = realize_force_constants(result, target)

    assert len(result.sparse[2].translations) == 7
    assert realized.materialize(2, max_bytes=None).shape == (1, 8, 3, 3)


def test_identifiability_accepts_resolved_and_rejects_folded_exact_interactions():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    space = build_primitive_interaction_space(
        primitive,
        order=2,
        cutoff=4.1,
        max_body_order=None,
        symprec=1e-5,
    )
    resolved = build_supercell(primitive, (3, 3, 3))
    validate_realization_identifiability(
        space, StructureRelation.from_atoms(primitive, resolved).index
    )

    folded = primitive.copy()
    with pytest.raises(InteractionAliasingError, match="larger single reference"):
        validate_realization_identifiability(
            space, StructureRelation.from_atoms(primitive, folded).index
        )


def test_identifiability_accepts_primitive_cell_with_irrational_cartesian_actions():
    """An fcc primitive cell rotates Cartesian tensor components by irrational entries."""
    primitive = bulk("Si", "diamond", a=5.43, cubic=False)
    space = build_primitive_interaction_space(
        primitive,
        order=2,
        cutoff=-1,
        max_body_order=None,
        symprec=1e-5,
    )
    index = StructureRelation.from_atoms(primitive, build_supercell(primitive, (2, 2, 2))).index
    validate_realization_identifiability(space, index)

    realized = realize_interaction_space(space, index)
    assert len(realized.orbits) == len(space.orbits)
    assert all(orbit.images for orbit in realized.orbits)
    frame = scaled_to_cartesian_matrix(realized.cell, realized.order)
    for orbit in realized.orbits:
        assert _observed_block_is_independent(frame, orbit)

    folded = StructureRelation.from_atoms(primitive, primitive.copy()).index
    with pytest.raises(InteractionAliasingError, match="larger single reference"):
        validate_realization_identifiability(space, folded)


@pytest.mark.parametrize(
    "primitive",
    [
        Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True),
        bulk("Si", "diamond", a=5.43, cubic=False),
        Atoms(
            "AlN",
            scaled_positions=[[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5]],
            cell=[[3.11, 0.0, 0.0], [-1.555, 2.693, 0.0], [0.0, 0.0, 4.98]],
            pbc=True,
        ),
    ],
)
def test_orbit_basis_is_integer_exact_and_renders_to_cartesian(primitive):
    r"""The lattice basis is integer, exactly invariant, and renders to Cartesian."""
    space = build_primitive_interaction_space(
        primitive,
        order=2,
        cutoff=4.0,
        max_body_order=None,
        symprec=1e-5,
    )
    frame = scaled_to_cartesian_matrix(np.asarray(primitive.cell.array, dtype=float), 2)
    assert space.orbits
    for orbit in space.orbits:
        lattice = np.asarray(orbit.basis)
        assert lattice.dtype == np.int64
        assert lattice.shape == (9, orbit.dimension)
        assert _observed_block_is_independent(frame, orbit)
        for image in orbit.images:
            if image.key != orbit.representative:
                continue
            np.testing.assert_array_equal(image.action.apply_scaled_columns(lattice), lattice)
            np.testing.assert_allclose(
                image.action.apply_columns(frame @ lattice),
                frame @ lattice,
                atol=1e-9,
            )


@pytest.mark.parametrize("order", [2, 3])
def test_generator_frames_describe_the_same_operation(order):
    r"""The lattice and Cartesian rotations of every generator are one operation.

    The two frames meet only through the cell map $K = (\text{cell}^T)^{\otimes k}$;
    this pins that pairing for a cell whose Cartesian rotations are irrational.
    """
    primitive = bulk("Si", "diamond", a=5.43, cubic=False)
    symmetry = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=1e-5)
    cell = np.asarray(primitive.cell.array, dtype=float)
    generators, _group = primitive_generators(symmetry, order, cell=cell)
    frame = scaled_to_cartesian_matrix(cell, order)
    inverse_frame = np.linalg.inv(frame)
    for generator in generators:
        action = generator.action
        assert action.scaled_rotation is not None
        np.testing.assert_allclose(
            action.as_matrix(),
            frame @ action.as_scaled_matrix() @ inverse_frame,
            atol=1e-9,
        )
        if generator.operation is not None:
            np.testing.assert_allclose(
                action.rotation,
                symmetry.cartesian_rotations[generator.operation].T,
                atol=1e-12,
            )


def test_identifiability_separates_folded_from_resolved_irrational_frames():
    """A too small reference is rejected in a frame with irrational Cartesian rotations."""
    primitive = bulk("Si", "diamond", a=5.43, cubic=False)
    space = build_primitive_interaction_space(
        primitive,
        order=2,
        cutoff=5.0,
        max_body_order=None,
        symprec=1e-5,
    )
    folded = StructureRelation.from_atoms(primitive, build_supercell(primitive, (2, 2, 2))).index
    with pytest.raises(InteractionAliasingError, match="larger single reference"):
        validate_realization_identifiability(space, folded)

    resolved = StructureRelation.from_atoms(primitive, build_supercell(primitive, (3, 3, 3))).index
    validate_realization_identifiability(space, resolved)


def test_exact_fc2_realization_into_sheared_supercell_matches_residue_mapping():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    source = build_supercell(primitive, (3, 3, 3))
    source_relation = StructureRelation.from_atoms(primitive, source)
    translations = np.asarray(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [2, -1, 3]], dtype=np.int32
    )
    tensors = np.asarray([(location + 1) * np.eye(3) for location in range(len(translations))])
    force_constants = ForceConstants(
        arrays={},
        supercell=source,
        sparse={
            2: SparseOrderForceConstants(
                2,
                np.zeros((len(translations), 2), dtype=np.int32),
                translations[:, None, :],
                tensors,
            )
        },
        relation=source_relation,
    )
    matrix = np.asarray([[2, 1, 0], [0, 2, 1], [0, 0, 2]], dtype=np.int32)
    target = build_supercell(primitive, matrix)
    relation = StructureRelation.from_atoms(primitive, target)

    actual = realize_force_constants(force_constants, target).materialize(2, max_bytes=None)
    expected = np.zeros((1, len(target), 3, 3))
    for translation, tensor in zip(translations, tensors, strict=True):
        matches = [
            atom
            for atom, candidate in enumerate(relation.cell_translation)
            if same_residue(candidate, translation, matrix)
        ]
        assert len(matches) == 1
        expected[0, matches[0]] += tensor
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def _fraction_free_rank(rows) -> int:
    """Reference exact rank by fraction-free elimination, used to check the certificate."""
    matrix = [list(map(int, row)) for row in np.atleast_2d(rows)]
    if not matrix or not matrix[0]:
        return 0
    n_rows, n_columns, rank = len(matrix), len(matrix[0]), 0
    for column in range(n_columns):
        pivot_row = next((row for row in range(rank, n_rows) if matrix[row][column]), None)
        if pivot_row is None:
            continue
        matrix[rank], matrix[pivot_row] = matrix[pivot_row], matrix[rank]
        pivot = matrix[rank][column]
        for row in range(rank + 1, n_rows):
            value = matrix[row][column]
            if value:
                matrix[row] = [
                    pivot * entry - value * reference
                    for entry, reference in zip(matrix[row], matrix[rank], strict=True)
                ]
        rank += 1
        if rank == n_rows:
            break
    return rank


def test_certified_rank_matches_fraction_free_elimination():
    """The modular certificate must agree with exact elimination, including bad primes."""
    rng = np.random.default_rng(5)
    for _ in range(60):
        shape = (int(rng.integers(1, 7)), int(rng.integers(1, 7)))
        matrix = rng.integers(-4, 5, size=shape)
        assert certified_rank(matrix) == _fraction_free_rank(matrix)

    prime, other = _RANK_PRIMES
    for matrix in (
        np.diag([prime, prime]),
        np.diag([prime * other, prime * other]),
        np.array([[prime * other, 1], [0, 1]]),
        np.array([[prime * other, 0], [1, 1]]),
    ):
        assert certified_rank(matrix) == _fraction_free_rank(matrix)
