import numpy as np
import pytest
from ase import Atoms

from mlfcs import SSCHA, FiniteDifferenceCalculation, realize_force_constants
from mlfcs.finite_difference.plan_identity import ForceBatch
from mlfcs.fitting.fitter import ForceConstantFitter
from mlfcs.force_constants.expansion import expand_primitive_parameters
from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants
from mlfcs.interactions.algebra.rendering import exact_lattice_coefficients
from mlfcs.interactions.keys import InteractionKey
from mlfcs.interactions.primitive.builder import build_primitive_interaction_space
from mlfcs.interactions.realization import (
    InteractionAliasingError,
    validate_realization_identifiability,
)
from mlfcs.interactions.space import InteractionSpace
from mlfcs.structure.integer_lattice import same_residue
from mlfcs.structure.relation import StructureRelation
from mlfcs.tools.supercell import build_supercell


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
def test_primitive_orbit_bases_are_invariant_and_orthonormal(order):
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    space = build_primitive_interaction_space(
        primitive,
        order=order,
        cutoff=4.1,
        max_body_order=2,
        symprec=1e-5,
    )

    for orbit in space.orbits:
        basis = orbit.cartesian_basis
        exact = orbit.exact_lattice_basis
        assert orbit.dimension == basis.shape[1]
        np.testing.assert_allclose(basis.T @ basis, np.eye(orbit.dimension), rtol=1e-10, atol=1e-10)
        assert exact.dtype == np.int64 and exact.shape == basis.shape
        # The two bases describe one subspace: C = K_n B_Z = Q R.
        np.testing.assert_allclose(
            space.frame.tensor_frame(order) @ exact,
            basis @ orbit.coefficient_transform,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            orbit.coefficient_transform,
            np.triu(orbit.coefficient_transform),
            rtol=0.0,
            atol=0.0,
        )
        for image in orbit.images:
            if image.key == orbit.representative:
                # The representative's own action is a symmetry of the basis block.
                np.testing.assert_allclose(
                    image.action.apply_columns(basis),
                    basis,
                    rtol=1e-9,
                    atol=1e-9,
                )


@pytest.mark.parametrize("order", [2, 3, 4])
def test_orbit_expansion_renders_every_image_from_the_representative_tensor(order):
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    space = build_primitive_interaction_space(
        primitive,
        order=order,
        cutoff=4.1,
        max_body_order=2,
        symprec=1e-5,
    )
    rng = np.random.default_rng(11 + order)
    parameters = rng.normal(size=space.n_parameters)
    sparse = expand_primitive_parameters(space, parameters)

    rendered = {
        (
            tuple(int(site) for site in sites),
            tuple(tuple(int(v) for v in t) for t in translations),
        ): tensor
        for sites, translations, tensor in zip(
            sparse.sites, sparse.translations, sparse.tensors, strict=True
        )
    }
    assert len(rendered) == len(sparse.sites)

    offset = 0
    shape = (3,) * order
    for orbit in space.orbits:
        coefficients = parameters[offset : offset + orbit.dimension]
        offset += orbit.dimension
        representative = orbit.cartesian_basis @ coefficients
        key = (orbit.representative.sites, orbit.representative.translations)
        np.testing.assert_allclose(rendered[key], representative.reshape(shape), atol=1e-12)
        for image in orbit.images:
            np.testing.assert_allclose(
                rendered[(image.key.sites, image.key.translations)],
                image.action.apply_flat(representative).reshape(shape),
                atol=1e-12,
            )


def test_observation_rows_reproduce_every_tensor_of_an_orbit():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    space = build_primitive_interaction_space(
        primitive,
        order=3,
        cutoff=4.1,
        max_body_order=2,
        symprec=1e-5,
    )

    for orbit in space.orbits:
        rng = np.random.default_rng(11 + orbit.dimension)
        parameters = rng.normal(size=orbit.dimension)
        tensor = orbit.cartesian_basis @ parameters
        observed = tensor[orbit.observation_rows]
        np.testing.assert_allclose(
            np.linalg.solve(orbit.observation_matrix, observed), parameters, rtol=1e-9, atol=1e-11
        )
        assert orbit.observation_matrix.shape == (orbit.dimension, orbit.dimension)
        assert orbit.observation_condition < 1e6


def test_exact_ifcs_realize_into_a_different_supercell_size():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    source = build_supercell(primitive, (3, 3, 3))
    calculation = FiniteDifferenceCalculation(
        primitive,
        reference=source,
        order=2,
        cutoff=4.1,
    )
    forces = np.zeros((len(calculation.plan), len(source), 3))
    result = calculation.reap(
        ForceBatch(
            fingerprint=calculation.manifest.fingerprint,
            configuration_ids=tuple(range(len(forces))),
            forces=forces,
        ),
        acoustic_sum_rule=False,
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
        space, StructureRelation.from_atoms(primitive, resolved, symprec=1e-5).index
    )

    folded = primitive.copy()
    with pytest.raises(InteractionAliasingError, match="larger single reference"):
        validate_realization_identifiability(
            space, StructureRelation.from_atoms(primitive, folded, symprec=1e-5).index
        )


def test_exact_fc2_realization_into_sheared_supercell_matches_residue_mapping():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    source = build_supercell(primitive, (3, 3, 3))
    source_relation = StructureRelation.from_atoms(primitive, source, symprec=1e-5)
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
    relation = StructureRelation.from_atoms(primitive, target, symprec=1e-5)

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


def test_export_site_mapping_uses_the_source_relations_symprec():
    """Export must not reintroduce a private site-mapping threshold."""
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.asarray([[4.0, 0.0, 0.0], [1.7, 4.2, 0.0], [0.8, 1.1, 4.5]]),
        pbc=True,
    )
    reference = build_supercell(primitive, (2, 1, 1), symprec=1e-3)
    relation = StructureRelation.from_atoms(primitive, reference, symprec=1e-3)
    force_constants = ForceConstants({}, reference, sparse={}, relation=relation)

    target_primitive = primitive.copy()
    target_primitive.positions[0, 0] += 5e-4
    target_reference = build_supercell(target_primitive, (2, 1, 1), symprec=1e-3)
    realized = realize_force_constants(
        force_constants, target_reference, primitive=target_primitive
    )
    assert realized.relation is not None
    assert realized.relation.symprec == 1e-3

    invalid_primitive = primitive.copy()
    invalid_primitive.positions[0, 0] += 1.1e-3
    invalid_reference = build_supercell(invalid_primitive, (2, 1, 1), symprec=1e-3)
    with pytest.raises(ValueError, match="site-mapping residual"):
        realize_force_constants(force_constants, invalid_reference, primitive=invalid_primitive)


def test_coefficient_transform_relates_cartesian_and_exact_lattice_coefficients():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    order = 3
    space = build_primitive_interaction_space(
        primitive,
        order=order,
        cutoff=4.1,
        max_body_order=2,
        symprec=1e-5,
    )
    frame = space.frame
    rng = np.random.default_rng(29)

    for orbit in space.orbits:
        parameters = rng.normal(size=orbit.dimension)
        cartesian = orbit.cartesian_basis @ parameters
        exact = exact_lattice_coefficients(orbit.coefficient_transform, parameters)
        np.testing.assert_allclose(
            frame.tensor_frame(order) @ (orbit.exact_lattice_basis @ exact),
            cartesian,
            atol=1e-10,
        )
        np.testing.assert_allclose(orbit.cartesian_basis.T @ cartesian, parameters, atol=1e-12)


def test_reference_resolved_cutoff_is_rejected():
    """`cutoff=None` is gone: the primitive model carries an explicit radius."""
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    reference = build_supercell(primitive, (2, 2, 2))

    with pytest.raises(ValueError, match="negative neighbour-shell"):
        FiniteDifferenceCalculation(primitive, reference=reference, order=2, cutoff=None)
    with pytest.raises(ValueError, match="reference-resolved cutoff=None"):
        ForceConstantFitter(primitive, reference, orders=(2,), cutoffs={2: None})
    with pytest.raises(ValueError, match="reference-resolved cutoff=None"):
        SSCHA(primitive, reference=reference, cutoff=None, snapshots=1, max_iterations=0)


def test_primitive_interaction_space_ignores_the_reference():
    """The primitive model is fixed by the primitive cell and an explicit radius.

    A reference that is too small to identify the model is a separate failure: the
    primitive space still exists, and identifiability rejects the reference instead of
    shortening the model.
    """
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    large = InteractionSpace(
        primitive, order=2, reference=build_supercell(primitive, (3, 3, 3)), cutoff=4.1
    )
    small = InteractionSpace(
        primitive, order=2, reference=build_supercell(primitive, (2, 2, 2)), cutoff=4.1
    )

    assert large.cutoff == small.cutoff == 4.1
    left, right = large.primitive_orbit_space, small.primitive_orbit_space
    assert len(left.orbits) == len(right.orbits)
    for first, second in zip(left.orbits, right.orbits, strict=True):
        assert first.representative == second.representative
        assert first.observation_rows.tolist() == second.observation_rows.tolist()
        np.testing.assert_array_equal(first.exact_lattice_basis, second.exact_lattice_basis)
        np.testing.assert_allclose(first.cartesian_basis, second.cartesian_basis, atol=1e-12)

    folded = InteractionSpace(primitive, order=2, reference=primitive.copy(), cutoff=4.1)
    assert len(folded.primitive_orbit_space.orbits) == len(left.orbits)
    with pytest.raises(InteractionAliasingError, match="larger single reference"):
        _ = folded.realized_orbit_space


def test_removed_orbit_fields_are_absent():
    """The integer-lattice refactor removed the ambiguous fields without aliases.

    `basis` mixed lattice, Cartesian and pivot-normalized coordinates, and `pivots` named
    observed components while standing in for parameters. Neither is restored as a
    compatibility alias, and the primitive space reports its lattice frame instead of the
    old source-frame `symmetry` field.
    """
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    space = build_primitive_interaction_space(
        primitive,
        order=2,
        cutoff=4.1,
        max_body_order=None,
        symprec=1e-5,
    )

    orbit = space.orbits[0]
    for name in ("basis", "pivots"):
        assert not hasattr(orbit, name), name
    assert not hasattr(space, "symmetry")
    assert orbit.exact_lattice_basis.shape[1] == orbit.dimension
    assert orbit.observation_rows.tolist() == sorted(orbit.observation_rows.tolist())
