import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from mlfcs.api import ForceConstantCalculation
from mlfcs.core.geometry import (
    PeriodicGeometry,
    StructureRelation,
    align_structures,
    make_supercell,
)


def test_relation_preserves_reference_order_for_a_nondiagonal_supercell():
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=[[3.0, 0, 0], [0.4, 3.2, 0], [0, 0, 4.0]],
        pbc=True,
    )
    matrix = np.asarray([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    generated, _ = make_supercell(primitive, matrix)
    permutation = np.asarray([3, 0, 6, 1, 7, 2, 5, 4])
    reference = generated[permutation]

    relation = StructureRelation.from_atoms(primitive, reference)

    np.testing.assert_array_equal(relation.reference.numbers, reference.numbers)
    np.testing.assert_array_equal(relation.supercell_matrix, matrix)
    assert relation.position_residual < 1e-10
    index = relation.index
    for atom in range(len(reference)):
        assert index.translate_atom(atom, [0, 0, 0]) == atom
        assert index.atom(index.primitive[atom], index.translations[atom]) == atom
    assert index.anchor((5, 7))[0] == index.representative(index.primitive[5])


def test_relation_maps_a_reordered_primitive_without_changing_reference_labels():
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=np.eye(3) * 4,
        pbc=True,
    )
    reference, _ = make_supercell(primitive, (2, 1, 1))
    reordered_primitive = primitive[[1, 0]]
    relation = StructureRelation.from_atoms(reordered_primitive, reference[[3, 0, 2, 1]])

    assert relation.index.n_primitive == 2
    np.testing.assert_array_equal(relation.reference.numbers, reference[[3, 0, 2, 1]].numbers)


def test_relation_uses_global_species_assignment_not_greedy_nearest_match():
    primitive = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=np.eye(3) * 10, pbc=True)
    # Atom 0 is equidistant from both sites; atom 1 can only use site 0.
    # A greedy argmin maps both to site 0, whereas the global optimum is
    # reference[0] -> site 1 and reference[1] -> site 0.
    reference = Atoms("H2", positions=[[0.5, 0, 0], [0, 0, 0]], cell=np.eye(3) * 10, pbc=True)
    relation = StructureRelation.from_atoms(primitive, reference, tolerance=0.6)

    np.testing.assert_array_equal(relation.primitive_index, [1, 0])


def test_align_structures_is_explicit_and_preserves_reference_order():
    reference = Atoms(
        "NaCl", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=np.eye(3) * 4, pbc=True
    )
    incoming = reference[[1, 0]]
    aligned, residual = align_structures(reference, incoming)

    assert residual < 1e-12
    np.testing.assert_array_equal(aligned.numbers, reference.numbers)
    np.testing.assert_allclose(aligned.positions, reference.positions)


def test_finite_difference_reap_is_invariant_to_reference_atom_permutation():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    generated, _ = make_supercell(primitive, [[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    reordered = generated[[1, 0]]
    canonical = ForceConstantCalculation(primitive, reference=generated, order=2, cutoff=3.0)
    shuffled = ForceConstantCalculation(primitive, reference=reordered, order=2, cutoff=3.0)

    # A deterministic reference-relative harmonic oracle supplies forces in
    # each calculation's own public atom order.
    forces_a = np.asarray([-(atoms.positions - generated.positions) for atoms in canonical.sow()])
    forces_b = np.asarray([-(atoms.positions - reordered.positions) for atoms in shuffled.sow()])
    fc_a = canonical.reap(forces_a, acoustic_sum_rule=False).sparse[2]
    fc_b = shuffled.reap(forces_b, acoustic_sum_rule=False).sparse[2]

    order_a = np.lexsort(
        (*fc_a.translation_representatives.reshape(len(fc_a.clusters), -1).T, *fc_a.sites.T)
    )
    order_b = np.lexsort(
        (*fc_b.translation_representatives.reshape(len(fc_b.clusters), -1).T, *fc_b.sites.T)
    )
    np.testing.assert_array_equal(fc_a.sites[order_a], fc_b.sites[order_b])
    np.testing.assert_array_equal(
        fc_a.translation_representatives[order_a], fc_b.translation_representatives[order_b]
    )
    np.testing.assert_allclose(fc_a.tensors[order_a], fc_b.tensors[order_b])


def test_reference_matrix_argument_is_an_explicit_consistency_assertion():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    reference, _ = make_supercell(primitive, (2, 1, 1))
    ForceConstantCalculation(
        primitive,
        reference=reference,
        supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        order=2,
        cutoff=3.0,
    )
    with np.testing.assert_raises_regex(ValueError, "disagree|does not match"):
        ForceConstantCalculation(
            primitive,
            reference=reference,
            supercell_matrix=[[1, 0, 0], [0, 2, 0], [0, 0, 1]],
            order=2,
            cutoff=3.0,
        )


def test_periodic_geometry_uses_general_mic_and_returns_degenerate_images():
    cell = np.asarray([[2.0, 0.0, 0.0], [1.9, 0.25, 0.0], [0.3, 0.1, 2.0]])
    geometry = PeriodicGeometry(cell)
    vector = np.asarray([1.1, 0.3, 0.0])
    expected, expected_length = find_mic(vector[None, :], cell, pbc=True)
    actual, shifts = geometry.closest_images(vector)

    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), expected_length[0])
    assert any(np.allclose(image, expected[0]) for image in actual)
    np.testing.assert_allclose(actual, vector + shifts @ cell)

    cubic = PeriodicGeometry(np.eye(3) * 2)
    images, shifts = cubic.closest_images(np.asarray([1.0, 0.0, 0.0]))
    assert len(images) == 2
    np.testing.assert_array_equal(np.sort(shifts[:, 0]), [-1, 0])
