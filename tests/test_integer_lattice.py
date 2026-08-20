import numpy as np

from mlfcs.core.integer_lattice import adjugate_3x3, determinant_3x3, residue_key, same_residue
from mlfcs.core.geometry import PeriodicIndex, _coset_translations, _translation_label


def test_integer_adjugate_is_exact_for_nondiagonal_signed_matrix():
    matrix = np.asarray([[2, -1, 1], [0, 3, 2], [1, 0, -2]], dtype=np.int64)
    determinant = determinant_3x3(matrix)
    adjugate = adjugate_3x3(matrix)

    np.testing.assert_array_equal(adjugate @ matrix, determinant * np.eye(3, dtype=np.int64))
    np.testing.assert_array_equal(matrix @ adjugate, determinant * np.eye(3, dtype=np.int64))


def test_residue_equivalence_is_exact_and_shared_by_core_components():
    matrix = np.asarray([[3, 1, 0], [-1, 2, 0], [0, 0, -2]], dtype=np.int64)
    translation = np.asarray([-7, 5, 4], dtype=np.int64)
    period = np.asarray([2, -3, 1], dtype=np.int64) @ matrix

    assert same_residue(translation, translation + period, matrix)
    assert not same_residue(translation, translation + [1, 0, 0], matrix)

    translations = _coset_translations(matrix)
    primitive = np.zeros(len(translations), dtype=np.int32)
    index = PeriodicIndex(primitive, translations, matrix)
    assert index.residue(translation) == residue_key(translation, matrix)
    assert _translation_label(translation, matrix) == residue_key(translation, matrix)


def test_residue_keys_have_exactly_determinant_many_representatives():
    matrix = np.asarray([[4, 1, 0], [1, 3, 0], [0, 0, -2]], dtype=np.int64)
    determinant = abs(determinant_3x3(matrix))
    keys = {residue_key([i, j, k], matrix) for i in range(12) for j in range(12) for k in range(4)}

    assert len(keys) == determinant
