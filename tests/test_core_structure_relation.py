from itertools import product

import numpy as np
import pytest
from ase import Atoms
from supercell_helpers import make_supercell

from mlfcs.finite_difference.calculation import FiniteDifferenceCalculation
from mlfcs.finite_difference.plan_identity import ForceBatch
from mlfcs.structure.periodic_geometry import PeriodicGeometry
from mlfcs.structure.relation import StructureRelation, align_structures


def _batch(job, forces):
    return ForceBatch(
        fingerprint=job.manifest.fingerprint,
        configuration_ids=tuple(range(len(forces))),
        forces=forces,
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
    assert relation.index is index
    for atom in range(len(reference)):
        assert index.translate_atom(atom, [0, 0, 0]) == atom
        assert index.atom(index.primitive[atom], index.translations[atom]) == atom
    atoms = np.arange(len(reference), dtype=np.int32)
    shifts = np.asarray([[0, 0, 0], [1, -2, 1], [-3, 0, 2]], dtype=np.int32)
    translated = index.translate_atoms(atoms, shifts)
    for shift_index, shift in enumerate(shifts):
        for atom in atoms:
            assert translated[shift_index, atom] == index.translate_atom(int(atom), shift)
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
    canonical = FiniteDifferenceCalculation(primitive, reference=generated, order=2, cutoff=3.0)
    shuffled = FiniteDifferenceCalculation(primitive, reference=reordered, order=2, cutoff=3.0)

    # A deterministic reference-relative harmonic oracle supplies forces in
    # each calculation's own public atom order.
    forces_a = np.asarray([-(atoms.positions - generated.positions) for atoms in canonical.sow()])
    forces_b = np.asarray([-(atoms.positions - reordered.positions) for atoms in shuffled.sow()])
    fc_a = canonical.reap(_batch(canonical, forces_a), acoustic_sum_rule=False).sparse[2]
    fc_b = shuffled.reap(_batch(shuffled, forces_b), acoustic_sum_rule=False).sparse[2]

    order_a = np.lexsort((*fc_a.translations.reshape(len(fc_a.tensors), -1).T, *fc_a.sites.T))
    order_b = np.lexsort((*fc_b.translations.reshape(len(fc_b.tensors), -1).T, *fc_b.sites.T))
    np.testing.assert_array_equal(fc_a.sites[order_a], fc_b.sites[order_b])
    np.testing.assert_array_equal(fc_a.translations[order_a], fc_b.translations[order_b])
    np.testing.assert_allclose(fc_a.tensors[order_a], fc_b.tensors[order_b])


def test_reference_matrix_argument_is_rejected_from_calculation_api():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    reference, _ = make_supercell(primitive, (2, 1, 1))
    with pytest.raises(TypeError, match="supercell_matrix"):
        FiniteDifferenceCalculation(
            primitive,
            reference=reference,
            supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
            order=2,
            cutoff=3.0,
        )
    with pytest.raises(TypeError, match="supercell_matrix"):
        FiniteDifferenceCalculation(
            primitive,
            reference=reference,
            supercell_matrix=[[1, 0, 0], [0, 2, 0], [0, 0, 1]],
            order=2,
            cutoff=3.0,
        )


def _brute_force_minimum_image(vector: np.ndarray, cell: np.ndarray):
    shifts = np.asarray(tuple(product(range(-3, 4), repeat=3)), dtype=np.int32)
    images = vector + shifts @ cell
    lengths = np.linalg.norm(images, axis=1)
    index = int(np.argmin(lengths))
    return images[index], float(lengths[index])


@pytest.mark.parametrize(
    "cell",
    [
        np.zeros((3, 3)),
        np.diag([1.0, 1.0, 0.0]),
        np.full((3, 3), np.nan),
        np.eye(2),
    ],
)
def test_periodic_geometry_rejects_invalid_cells_before_reduction(cell):
    with pytest.raises(ValueError, match="finite, nonsingular 3x3 cell"):
        PeriodicGeometry(cell)


def test_periodic_geometry_returns_degenerate_images_at_the_minimum_image():
    cell = np.asarray([[2.0, 0.0, 0.0], [1.9, 0.25, 0.0], [0.3, 0.1, 2.0]])
    geometry = PeriodicGeometry(cell)
    vector = np.asarray([1.1, 0.3, 0.0])
    expected, expected_length = _brute_force_minimum_image(vector, cell)
    actual, shifts = geometry.closest_images(vector)

    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), expected_length)
    assert any(np.allclose(image, expected) for image in actual)
    np.testing.assert_allclose(actual, vector + shifts @ cell)

    cubic = PeriodicGeometry(np.eye(3) * 2)
    images, shifts = cubic.closest_images(np.asarray([1.0, 0.0, 0.0]))
    assert len(images) == 2
    np.testing.assert_array_equal(np.sort(shifts[:, 0]), [-1, 0])


def test_periodic_geometry_mic_is_exact_for_a_skewed_cell():
    # ase.geometry.find_mic returns 0.1708800749 here: its fast path skips the
    # Minkowski reduction because the folded vector is shorter than
    # 0.5 * min(cell.lengths()) = 0.3, which is not a sufficient criterion.
    cell = np.asarray([[0.6, 0.0, 0.0], [-1.0, 0.1, 0.0], [-0.4, -0.5, 0.1]])
    geometry = PeriodicGeometry(cell)
    vector = np.asarray([0.12, 0.12, -0.02])

    expected, expected_length = _brute_force_minimum_image(vector, cell)
    minimum, length = geometry.mic(vector)

    np.testing.assert_allclose(expected, [-0.08, 0.02, -0.02])
    np.testing.assert_allclose(minimum, expected)
    np.testing.assert_allclose(float(length), expected_length)

    batch, lengths = geometry.mic(vector[None, :])
    np.testing.assert_allclose(batch[0], expected)
    np.testing.assert_allclose(lengths, expected_length)


def test_periodic_geometry_resolves_only_jointly_compatible_cluster_images():
    geometry = PeriodicGeometry(np.diag([4.0, 1.0, 1.0]))
    shifts = geometry.joint_closest_image_shifts(np.asarray([[1.0, 0, 0], [2.0, 0, 0]]))

    np.testing.assert_array_equal(shifts, np.asarray([[[0, 0, 0], [0, 0, 0]]]))

    impossible = PeriodicGeometry(np.diag([3.0, 1.0, 1.0]))
    shifts = impossible.joint_closest_image_shifts(np.asarray([[1.0, 0, 0], [2.0, 0, 0]]))
    assert shifts.shape == (0, 2, 3)
