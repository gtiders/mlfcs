import numpy as np
import pytest
from ase import Atoms

from mlfcs import ForceConstantCalculation, build_supercell, realize_force_constants
from mlfcs.force_constants.data import ForceConstants, SparseOrderForceConstants
from mlfcs.interactions.enumerate import (
    build_primitive_interaction_space,
)
from mlfcs.interactions.keys import InteractionKey
from mlfcs.interactions.realization import (
    InteractionAliasingError,
    validate_realization_identifiability,
)
from mlfcs.structure.integer_lattice import same_residue
from mlfcs.structure.relation import StructureRelation


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
def test_primitive_orbit_bases_are_invariant_and_pivot_normalized(order):
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    space = build_primitive_interaction_space(
        primitive,
        order=order,
        cutoff=4.1,
        max_body_order=2,
        symprec=1e-5,
    )

    for orbit in space.orbits:
        np.testing.assert_allclose(
            orbit.basis[orbit.pivots],
            np.eye(orbit.dimension),
            rtol=1e-10,
            atol=1e-10,
        )
        for image in orbit.images:
            if image.key == orbit.representative:
                np.testing.assert_allclose(
                    image.action.apply_columns(orbit.basis),
                    orbit.basis,
                    rtol=1e-9,
                    atol=1e-9,
                )


def test_exact_ifcs_realize_into_a_different_supercell_size():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    source = build_supercell(primitive, (3, 3, 3))
    calculation = ForceConstantCalculation(
        primitive,
        reference=source,
        order=2,
        cutoff=4.1,
        verbose=False,
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
