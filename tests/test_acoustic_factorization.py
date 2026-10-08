"""Independent exact rank, factor and physical-coordinate prototype checks."""

import numpy as np
import pytest
from scipy import sparse

from mlfcs.cluster_space.acoustic import prepare_coordinates
from mlfcs.cluster_space.acoustic_factorization import checked_add, checked_multiply


@pytest.mark.parametrize(
    "values",
    [
        [[2, 3, 1], [4, 6, 2]],
        [[3, 1, 2, 0], [0, 5, 1, 2], [3, 6, 3, 2]],
        [[1, 0], [0, 1]],
        [[0, 0, 0]],
        [[2, -3, 5, 1], [7, 2, 0, -4], [9, -1, 5, -3]],
    ],
)
def test_exact_factor_rank_and_span(values):
    """Compare certified factors against an arbitrary-precision test oracle."""
    Matrix = pytest.importorskip("sympy").Matrix
    matrix = np.asarray(values, dtype=np.int64)
    factor = prepare_coordinates(sparse.csr_matrix(matrix), (np.eye(matrix.shape[1]),))
    assert len(factor.pivots) == Matrix(matrix).rank()
    columns = [factor.lift(column) for column in np.eye(len(factor.free))]
    basis = np.asarray(columns).T if columns else np.empty((matrix.shape[1], 0))
    np.testing.assert_allclose(matrix @ basis, 0, atol=1e-13)
    assert np.linalg.matrix_rank(basis) == len(factor.free) if columns else basis.shape[1] == 0


def test_actual_word_overflow_is_rejected():
    """Ensure certificate arithmetic raises before a product or sum wraps."""
    maximum = np.iinfo(np.int64).max
    with pytest.raises(OverflowError):
        checked_multiply(np.int64(maximum), np.int64(2))
    with pytest.raises(OverflowError):
        checked_add(np.int64(maximum), np.int64(1))
    assert checked_multiply(np.int64(-7), np.int64(8)) == -56


def test_random_sparse_factors_against_rational_oracle():
    """Certify ranks and real nullspaces on deterministic small sparse matrices."""
    Matrix = pytest.importorskip("sympy").Matrix
    rng = np.random.default_rng(175)
    for _ in range(30):
        matrix = rng.integers(-5, 6, size=(8, 11), dtype=np.int64)
        matrix[rng.random(matrix.shape) < 0.7] = 0
        matrix[-1] = matrix[0] + matrix[1]
        factor = prepare_coordinates(sparse.csr_matrix(matrix), (np.eye(matrix.shape[1]),))
        assert len(factor.pivots) == Matrix(matrix).rank()
        free = rng.normal(size=factor.dimension)
        np.testing.assert_allclose(matrix @ factor.lift(free), 0, atol=1e-12)


@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (3, 0)])
def test_empty_constraint_spaces(shape):
    """Certify empty equation and parameter spaces without inventing constraints."""
    factor = prepare_coordinates(sparse.csr_matrix(shape, dtype=np.int64), (np.eye(shape[1]),))
    assert len(factor.pivots) == 0
    assert factor.dimension == shape[1]
    np.testing.assert_array_equal(factor.lift(np.ones(shape[1])), np.ones(shape[1]))


def test_modular_zero_does_not_drop_an_integer_coefficient():
    """Recover factors whose support differs between the two residue fields."""
    prime = 2147483647
    matrix = np.array([[1, prime, 2], [0, 1, 3]], dtype=np.int64)
    factor = prepare_coordinates(sparse.csr_matrix(matrix), (np.eye(3),))
    assert factor.dimension == 1
    np.testing.assert_array_equal(factor.pivots, [1, 0])
    np.testing.assert_array_equal(factor.lift(np.ones(1)), [3 * prime - 2, -3, 1])
    np.testing.assert_array_equal(matrix @ factor.lift(np.ones(1)), np.zeros(2))


def test_certificate_rejects_a_corrupted_compressed_factor():
    """The characteristic-zero identity must reject an incorrect reconstructed entry."""
    from mlfcs.cluster_space.acoustic_factorization import _certify

    matrix = sparse.csr_matrix(np.array([[2, 3, 1], [4, 6, 2]], dtype=np.int64))
    upper = (np.array([0, 3]), np.array([0, 1, 2]), np.array([2, 3, 1]))
    lower = (np.array([0, 1, 2]), np.array([0, 0]), np.array([2, 4]))
    denominators = np.array([2])
    assert _certify(
        matrix.indptr,
        matrix.indices,
        matrix.data,
        upper,
        denominators,
        lower,
        lower,
        np.array([0]),
        1,
    )
    upper[2][1] += 1
    assert not _certify(
        matrix.indptr,
        matrix.indices,
        matrix.data,
        upper,
        denominators,
        lower,
        lower,
        np.array([0]),
        1,
    )
