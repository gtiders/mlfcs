from itertools import product

import numpy as np
import pytest

from mlfcs.core.geometry import PeriodicIndex, _coset_translations, _translation_label
from mlfcs.core.integer_lattice import (
    IntegerLatticeQuotient,
    adjugate_3x3,
    determinant_3x3,
    residue_key,
    row_hermite_normal_form,
    same_residue,
)


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
    assert index.residue(translation) == index.residue(translation + period)
    assert _translation_label(translation, matrix) == residue_key(translation, matrix)


def test_residue_keys_have_exactly_determinant_many_representatives():
    matrix = np.asarray([[4, 1, 0], [1, 3, 0], [0, 0, -2]], dtype=np.int64)
    determinant = abs(determinant_3x3(matrix))
    keys = {residue_key([i, j, k], matrix) for i in range(12) for j in range(12) for k in range(4)}

    assert len(keys) == determinant


@pytest.mark.parametrize(
    "matrix",
    [
        [[2, 0, 0], [0, 3, 0], [0, 0, 1]],
        [[2, 1, 0], [0, 2, 1], [0, 0, 1]],
        [[1, 2, 0], [2, 0, 0], [0, 0, 1]],
        [[-2, 1, 0], [0, 2, 1], [0, 0, 1]],
        [[3, -2, 1], [1, 1, 0], [0, 1, 2]],
    ],
)
def test_row_hnf_is_exact_canonical_and_unimodular(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    hnf, transform = row_hermite_normal_form(matrix)

    np.testing.assert_array_equal(hnf, transform @ matrix)
    assert abs(determinant_3x3(transform)) == 1
    assert abs(determinant_3x3(hnf)) == abs(determinant_3x3(matrix))
    assert np.all(np.diag(hnf) > 0)
    assert np.all(np.triu(hnf, 1) == 0)


@pytest.mark.parametrize(
    "matrix",
    [
        [[2, 0, 0], [0, 3, 0], [0, 0, 1]],
        [[2, 1, 0], [0, 2, 1], [0, 0, 1]],
        [[1, 2, 0], [2, 0, 0], [0, 0, 1]],
        [[-2, 1, 0], [0, 2, 1], [0, 0, 1]],
        [[3, -2, 1], [1, 1, 0], [0, 1, 2]],
    ],
)
def test_hnf_fundamental_domain_matches_exact_residue_equivalence(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    quotient = IntegerLatticeQuotient(matrix)
    determinant = abs(determinant_3x3(matrix))

    assert quotient.size == determinant
    assert len({residue_key(value, matrix) for value in quotient.representatives}) == determinant
    for translation in product(range(-4, 5), repeat=3):
        q, remainder = quotient.decompose(np.asarray(translation, dtype=np.int64))
        np.testing.assert_array_equal(q @ quotient.hnf + remainder, translation)
        assert quotient.cell_index(translation) == quotient.cell_index(remainder)
        assert quotient.equivalent(translation, remainder)
        assert same_residue(translation, remainder, matrix)


def test_hnf_is_invariant_under_unimodular_row_basis_changes():
    matrix = np.asarray([[3, 1, 0], [0, 2, 1], [0, 0, 2]], dtype=np.int64)
    change = np.asarray([[1, -2, 1], [0, 1, -1], [0, 0, 1]], dtype=np.int64)

    left = IntegerLatticeQuotient(matrix)
    right = IntegerLatticeQuotient(change @ matrix)

    np.testing.assert_array_equal(left.hnf, right.hnf)
    np.testing.assert_array_equal(left.representatives, right.representatives)
