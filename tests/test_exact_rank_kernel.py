"""Exact rank certification and saturated integer kernels.

The oracle throughout is ``sympy.Matrix(...).rank()`` (exact rational rank); the
modular rank has its own oracle in Galois-field ``rref`` pivots, because the
rational rank of a matrix whose entries are already reduced modulo a prime can
exceed its rank over that prime.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import sympy
from sympy import GF, Matrix
from sympy.polys.matrices import DomainMatrix

from mlfcs.exceptions import IntegerRangeError, RankCertificateError
from mlfcs.interactions.algebra.actions import TensorAction
from mlfcs.interactions.algebra.exact import (
    RANK_PRIMES,
    canonical_lattice_basis,
    certified_pivots,
    certified_rank,
    hadamard_bound,
    modular_echelon,
    modular_rank,
    prime_stream,
    saturated_kernel,
    to_python_rows,
    verify_kernel,
)
from mlfcs.interactions.algebra.invariants import invariant_kernel, label_symmetric_basis

SEED = 20240921
_BOX = 3
_COEFFICIENT_BOUND = 12
_PRIME_PREFIX = (
    2147483647,
    2147483629,
    2147483587,
    2147483579,
    2147483563,
    2147483549,
    2147483543,
    2147483497,
)

_RANK_CASES = {
    "zero_square": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    "zero_wide": [[0, 0, 0]],
    "empty_list": [],
    "empty_0x0": np.zeros((0, 0), dtype=np.int64),
    "empty_0x3": np.zeros((0, 3), dtype=np.int64),
    "empty_3x0": np.zeros((3, 0), dtype=np.int64),
    "rowless_columns": [[], []],
    "wide_rank_one": [[1, 2, 3, 4], [2, 4, 6, 8]],
    "wide_full": [[1, 0, 0, 0], [0, 1, 0, 0]],
    "tall_full": [[1, 0], [0, 1], [1, 1]],
    "full_square": [[2, 1], [1, 3]],
    "deficient_square": [[1, 2, 3], [2, 4, 6], [3, 6, 9]],
    "deficient_3x3_adjacent_columns": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    "duplicate_rows": [[1, 2], [3, 4], [1, 2]],
    "duplicate_columns": [[1, 1, 2], [3, 3, 4]],
    "identity_int32": np.eye(4, dtype=np.int32),
    "integer_uint8": np.array([[1, 2], [2, 4]], dtype=np.uint8),
}

_SMALL_KERNEL_CASES = {
    "wide_rank_one": ([[1, 2, 3, 4], [2, 4, 6, 8]], 4),
    "tall": ([[1, 0], [0, 1], [1, 1]], 2),
    "hidden_vector": ([[1, -2, 1]], 3),
    "dyadic": ([[2, 1, 1]], 3),
    "zero_square": ([[0, 0], [0, 0]], 2),
    "duplicate_rows": ([[1, 1, 2], [1, 1, 2]], 3),
    "no_columns": ([[], []], 0),
    "no_rows": (np.zeros((0, 3), dtype=np.int64), 3),
    "partial_identity": ([[1, 0, 0], [0, 1, 0]], 3),
    "full_square": ([[1, 2], [3, 4]], 2),
    "empty": (np.zeros((0, 0), dtype=np.int64), 0),
}


def _rows(matrix) -> list[list[int]]:
    """Rows of a test case, independent of the module under test."""
    if isinstance(matrix, np.ndarray):
        return matrix.tolist()
    return [list(row) for row in matrix]


def _shape(matrix) -> tuple[int, int]:
    if isinstance(matrix, np.ndarray):
        return int(matrix.shape[0]), int(matrix.shape[1])
    rows = _rows(matrix)
    return (len(rows), len(rows[0])) if rows else (0, 0)


def _oracle_rank(matrix) -> int:
    return int(Matrix(_rows(matrix)).rank())


def _oracle_modular_rank(matrix, prime: int) -> int:
    domain = DomainMatrix.from_Matrix(Matrix(_rows(matrix))).convert_to(GF(prime))
    return len(domain.rref()[1])


def _sympy_matrix(matrix, width: int) -> Matrix:
    """SymPy view of a case, keeping the shape of row-less matrices."""
    rows = _rows(matrix)
    return Matrix(rows) if rows else Matrix.zeros(0, width)


def _integer_kernel_vectors(matrix, width: int, bound: int = _BOX) -> set[tuple[int, ...]]:
    """Brute-force ``{x in the box : matrix @ x == 0}`` over the integers."""
    rows = _rows(matrix)
    return {
        point
        for point in itertools.product(range(-bound, bound + 1), repeat=width)
        if all(sum(value * entry for value, entry in zip(row, point)) == 0 for row in rows)
    }


def _lattice_points(
    basis: np.ndarray,
    width: int,
    bound: int = _BOX,
    coefficient_bound: int | None = None,
) -> set[tuple[int, ...]]:
    """Box points reached by integer combinations of the basis columns."""
    basis_rows = basis.tolist()
    columns = len(basis_rows[0]) if width else 0
    if coefficient_bound is None:
        coefficient_bound = _COEFFICIENT_BOUND if columns <= 3 else 6
    points = set()
    for coefficients in itertools.product(
        range(-coefficient_bound, coefficient_bound + 1), repeat=columns
    ):
        point = tuple(
            sum(value * coefficient for value, coefficient in zip(row, coefficients))
            for row in basis_rows
        )
        if all(abs(value) <= bound for value in point):
            points.add(point)
    return points


def _integer_coordinates(basis: np.ndarray, vector) -> list[int] | None:
    """Exact coordinates of ``vector`` in the lattice spanned by the basis columns."""
    solution = Matrix(basis.tolist()).gauss_jordan_solve(Matrix(list(vector)))[0]
    if any(value.q != 1 for value in solution):
        return None
    return [int(value) for value in solution]


def _assert_pivots_are_valid(matrix, rank, pivot_rows, pivot_columns) -> None:
    height, width = _shape(matrix)
    assert isinstance(pivot_rows, np.ndarray) and pivot_rows.dtype == np.int64
    assert isinstance(pivot_columns, np.ndarray) and pivot_columns.dtype == np.int64
    assert pivot_rows.size == pivot_columns.size == rank
    assert len(set(pivot_rows.tolist())) == rank
    assert len(set(pivot_columns.tolist())) == rank
    assert all(0 <= int(index) < height for index in pivot_rows)
    assert all(0 <= int(index) < width for index in pivot_columns)
    if rank == 0:
        return
    rows = _rows(matrix)
    selected = [rows[int(index)] for index in pivot_rows]
    assert Matrix(selected).rank() == rank
    # The selected rows span the row space of the whole matrix over the rationals.
    assert Matrix(selected + rows).rank() == rank


@pytest.mark.parametrize("name", sorted(_RANK_CASES))
def test_rank_matches_sympy_oracle(name):
    matrix = _RANK_CASES[name]

    rank, pivot_rows, pivot_columns = certified_pivots(matrix)

    assert rank == certified_rank(matrix)
    assert rank == _oracle_rank(matrix)
    _assert_pivots_are_valid(matrix, rank, pivot_rows, pivot_columns)


def test_rank_survives_primes_that_divide_the_witness_minor():
    prime_a, prime_b = RANK_PRIMES
    matrix = [[prime_a, 0, 0], [0, prime_b, 0], [prime_a, prime_b, 0]]

    # The 2x2 minor [[prime_a, 0], [0, prime_b]] vanishes modulo both fast primes,
    # so the first two primes only see rank 1; the certificate needs a third.
    assert modular_rank(matrix, prime_a) == 1
    assert modular_rank(matrix, prime_b) == 1
    assert modular_rank(matrix, _PRIME_PREFIX[2]) == 2
    assert hadamard_bound(matrix, 2) > prime_a * prime_b

    rank, pivot_rows, pivot_columns = certified_pivots(matrix)
    assert rank == certified_rank(matrix) == _oracle_rank(matrix) == 2
    _assert_pivots_are_valid(matrix, rank, pivot_rows, pivot_columns)

    scaled = [[prime_a * prime_b, 0], [0, 1]]
    assert modular_rank(scaled, prime_a) == 1
    assert certified_rank(scaled) == _oracle_rank(scaled) == 2


def test_rank_matches_oracle_on_random_matrices():
    rng = np.random.default_rng(SEED)
    for height, width in ((1, 5), (5, 1), (3, 3), (4, 6), (6, 4), (7, 7), (2, 9), (9, 2)):
        for _ in range(3):
            matrix = rng.integers(-9, 10, size=(height, width), dtype=np.int64)
            rank, pivot_rows, pivot_columns = certified_pivots(matrix)
            assert rank == certified_rank(matrix) == _oracle_rank(matrix)
            _assert_pivots_are_valid(matrix, rank, pivot_rows, pivot_columns)

        factors = rng.integers(-4, 5, size=(height, 2), dtype=np.int64)
        multipliers = rng.integers(-4, 5, size=(2, width), dtype=np.int64)
        deficient = factors @ multipliers
        rank, pivot_rows, pivot_columns = certified_pivots(deficient)
        assert rank == certified_rank(deficient) == _oracle_rank(deficient)
        _assert_pivots_are_valid(deficient, rank, pivot_rows, pivot_columns)


def test_rank_of_large_magnitudes_is_exact_and_never_overflows():
    cases = {
        "int64_max": np.array([[2**63 - 1, 0], [0, 2**63 - 1], [1, 1]], dtype=np.int64),
        "two_to_63": [[2**63, 0], [0, 2**63], [1, 1]],
        "ten_to_30_off_by_one": [[10**30, 10**30 + 1], [10**30 + 1, 10**30]],
        "negative_ten_to_18": [[-(10**18), 1], [1, 0]],
        "object_rows": np.array([[10**30, 2**63], [-(10**18), 3**40]], dtype=object),
    }

    for name, matrix in cases.items():
        rank, pivot_rows, pivot_columns = certified_pivots(matrix)
        assert rank == certified_rank(matrix) == _oracle_rank(matrix), name
        _assert_pivots_are_valid(matrix, rank, pivot_rows, pivot_columns)

    # Collapsing these entries to float64 loses the off-by-one and would report rank 1.
    assert certified_rank([[10**30, 10**30 + 1], [10**30 + 1, 10**30]]) == 2


def test_rank_of_sparse_high_dynamic_range_matrix():
    matrix = np.zeros((5, 5), dtype=object)
    matrix[0, 0] = 2**63
    matrix[1, 1] = 10**30
    matrix[2, 2] = -(10**18)
    matrix[3, 3] = 7
    matrix[4, 0] = 3**40
    matrix[4, 3] = 5**25

    rank, pivot_rows, pivot_columns = certified_pivots(matrix)
    assert rank == certified_rank(matrix) == _oracle_rank(matrix) == 4
    _assert_pivots_are_valid(matrix, rank, pivot_rows, pivot_columns)


def test_certified_rank_is_a_thin_wrapper_over_certified_pivots():
    rng = np.random.default_rng(SEED + 1)
    for _ in range(4):
        matrix = rng.integers(-30, 31, size=(5, 4), dtype=np.int64)
        assert certified_rank(matrix) == certified_pivots(matrix)[0]
    assert certified_rank([]) == certified_pivots([])[0] == 0
    assert certified_rank(np.zeros((0, 4), dtype=np.int64)) == 0
    assert certified_rank(np.zeros((4, 0), dtype=np.int64)) == 0


def test_certified_pivots_row_selection_reproduces_the_integer_kernel():
    cases = [
        ([[1, 2, 3, 4], [2, 4, 6, 8]], 4),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3),
        ([[0, 0], [1, 0], [0, 1]], 2),
        ([[RANK_PRIMES[0], 0, 0]], 3),
        (np.zeros((0, 3), dtype=np.int64), 3),
    ]

    for matrix, width in cases:
        rank, pivot_rows, pivot_columns = certified_pivots(matrix)
        assert rank == _oracle_rank(matrix)
        _assert_pivots_are_valid(matrix, rank, pivot_rows, pivot_columns)
        selected = [_rows(matrix)[int(index)] for index in pivot_rows]
        assert _integer_kernel_vectors(selected, width, bound=2) == _integer_kernel_vectors(
            matrix, width, bound=2
        )


def test_rank_certificate_error_when_the_prime_stream_is_exhausted(monkeypatch):
    matrix = [[RANK_PRIMES[0], 0, 0], [0, RANK_PRIMES[1], 0], [RANK_PRIMES[0], RANK_PRIMES[1], 0]]
    monkeypatch.setattr(
        "mlfcs.interactions.algebra.exact.prime_stream", lambda: iter([RANK_PRIMES[0]])
    )

    with pytest.raises(RankCertificateError):
        certified_rank(matrix)


def test_prime_stream_starts_with_the_rank_primes_and_descends():
    stream = prime_stream()
    entries = [next(stream) for _ in range(60)]

    assert tuple(entries[:2]) == RANK_PRIMES
    assert tuple(entries[:8]) == _PRIME_PREFIX
    assert len(set(entries)) == 60
    assert all(later < earlier for earlier, later in itertools.pairwise(entries))
    assert all(sympy.isprime(entry) for entry in entries)
    assert entries[-1] < 2**31
    assert all(entry < 2**31 for entry in entries)


def test_modular_rank_matches_galois_field_rank_and_bounds_the_exact_rank():
    rng = np.random.default_rng(SEED + 2)
    for _ in range(6):
        matrix = rng.integers(-20, 21, size=(4, 5), dtype=np.int64)
        for prime in RANK_PRIMES:
            assert modular_rank(matrix, prime) == _oracle_modular_rank(matrix, prime)
            assert modular_rank(matrix, prime) <= _oracle_rank(matrix)

    # Entries reduced modulo a prime can still be rationally independent.
    collapsing = [[4, 1], [1, 2]]
    assert _oracle_rank(collapsing) == 2
    assert modular_rank(collapsing, 7) == _oracle_modular_rank(collapsing, 7) == 1


def test_modular_echelon_pivots_reference_the_input_row_order():
    matrix = [[0, 0], [1, 0], [0, 1]]

    echelon, pivot_columns, pivot_rows = modular_echelon(matrix, RANK_PRIMES[0])

    assert echelon.dtype == np.int64
    assert echelon.shape == (3, 2)
    assert pivot_columns.tolist() == [0, 1]
    assert pivot_rows.tolist() == [1, 2]
    for position, column in enumerate(pivot_columns.tolist()):
        assert echelon[position, column] == 1
        assert not echelon[position + 1 :, column].any()

    with pytest.raises(ValueError):
        modular_echelon(matrix, 1)


def test_hadamard_bound_is_an_exact_integer_upper_bound():
    matrix = [[3, 4], [1, -7], [2, 2]]
    ranks = list(itertools.combinations(range(3), 2))
    columns = list(itertools.combinations(range(2), 2))
    minors = [
        abs(
            Matrix(
                [
                    [matrix[top][left], matrix[top][right]],
                    [matrix[bottom][left], matrix[bottom][right]],
                ]
            ).det()
        )
        for top, bottom in ranks
        for left, right in columns
    ]

    bound = hadamard_bound(matrix, 2)

    assert isinstance(bound, int)
    assert max(minors) < bound
    # Row norms are 6, 8 and 3, so the widest single row bound is 8.
    assert hadamard_bound(matrix, 1) == 8
    assert hadamard_bound(matrix, 2) == 48
    assert hadamard_bound([[3, 4]], 1) == 6
    assert hadamard_bound([[2**63, 0]], 1) == 2**63 + 1
    assert hadamard_bound(matrix, 0) == 1
    assert hadamard_bound(matrix, -3) == 1
    assert hadamard_bound([], 0) == 1


@pytest.mark.parametrize("name", sorted(_SMALL_KERNEL_CASES))
def test_saturated_kernel_is_exact_saturated_and_full_rank(name):
    matrix, width = _SMALL_KERNEL_CASES[name]
    rank = _oracle_rank(matrix)

    basis = saturated_kernel(matrix)

    assert basis.dtype == np.int64
    assert basis.shape == (width, width - rank)
    verify_kernel(matrix, basis)
    assert int(Matrix(basis.tolist()).rank()) == width - rank

    nullspace = _sympy_matrix(matrix, width).nullspace()
    assert len(nullspace) == width - rank
    if rank < width:
        assert Matrix.hstack(Matrix(basis.tolist()), *nullspace).rank() == width - rank

    box = _integer_kernel_vectors(matrix, width)
    reached = _lattice_points(basis, width)
    assert reached <= box  # basis columns are kernel vectors
    assert box <= reached  # nothing in the integer kernel is missed: saturation


def test_saturated_kernel_reaches_an_integer_kernel_vector_rational_bases_miss():
    matrix = [[1, -2, 1]]

    basis = saturated_kernel(matrix)

    verify_kernel(matrix, basis)
    assert basis.shape == (3, 2)
    # A primitive rational-nullspace basis need not generate (1, 1, 1) ... but this
    # one does, so the vector is reachable with integral coordinates.
    coordinates = _integer_coordinates(basis, (1, 1, 1))
    assert coordinates is not None
    assert (Matrix(basis.tolist()) @ Matrix(coordinates)).T == Matrix([[1, 1, 1]])


def test_saturated_kernel_beats_a_primitive_but_unsaturated_basis():
    matrix = [[2, 1, 1]]
    unsaturated = np.array([[-1, -1], [2, 0], [0, 2]], dtype=np.int64)

    # The primitive columns above are a rational basis of the kernel but generate
    # only a sublattice of it: (1, 1, -3) is an integer kernel vector they miss.
    assert _oracle_rank(np.array([[-1], [2], [0]])) == 1
    verify_kernel(matrix, unsaturated)
    assert _integer_coordinates(unsaturated, (1, 1, -3)) is None

    basis = saturated_kernel(matrix)

    verify_kernel(matrix, basis)
    coordinates = _integer_coordinates(basis, (1, 1, -3))
    assert coordinates is not None
    assert (Matrix(basis.tolist()) @ Matrix(coordinates)).T == Matrix([[1, 1, -3]])
    assert (1, 1, -3) not in _lattice_points(unsaturated, 3)
    assert (1, 1, -3) in _lattice_points(basis, 3)


def test_saturated_kernel_degenerate_shapes():
    np.testing.assert_array_equal(
        saturated_kernel(np.zeros((0, 3), dtype=np.int64)), np.eye(3, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        saturated_kernel(np.zeros((0, 0), dtype=np.int64)), np.zeros((0, 0), dtype=np.int64)
    )
    assert saturated_kernel([]).shape == (0, 0)
    assert saturated_kernel(np.zeros((3, 0), dtype=np.int64)).shape == (0, 0)
    assert saturated_kernel([[1, 2], [3, 4]]).shape == (2, 0)

    verify_kernel([], [])
    verify_kernel(np.zeros((0, 3), dtype=np.int64), np.eye(3, dtype=np.int64))
    verify_kernel([[1, 2], [3, 4]], saturated_kernel([[1, 2], [3, 4]]))


def test_saturated_kernel_raises_for_entries_that_do_not_fit_int64():
    # A 2**70 entry forces a 2**70 kernel direction, which int64 cannot carry.
    with pytest.raises(IntegerRangeError):
        saturated_kernel([[2**70, 1]])
    with pytest.raises(IntegerRangeError):
        saturated_kernel([[2**70, 0, 1]])

    # The literal row [2**70, 0] has the small kernel direction (0, 1).
    np.testing.assert_array_equal(
        saturated_kernel([[2**70, 0]]), np.array([[0], [1]], dtype=np.int64)
    )


def test_verify_kernel_rejects_a_wrong_basis_and_ids_the_failure():
    matrix = [[1, 2], [3, 4]]

    with pytest.raises(ValueError, match=r"row 0, column 0.*residual 1"):
        verify_kernel(matrix, np.eye(2, dtype=np.int64))
    with pytest.raises(ValueError, match="columns"):
        verify_kernel(matrix, np.zeros((3, 1), dtype=np.int64))
    # Exact integer arithmetic: an int64 product would wrap 2**64 to zero here.
    with pytest.raises(ValueError, match=r"residual 18446744073709551616"):
        verify_kernel([[2**63, 2**63]], np.ones((2, 1), dtype=np.int64))

    verify_kernel([[2**63, 2**63]], np.array([[1], [-1]], dtype=np.int64))


def test_to_python_rows_accepts_every_integer_representation():
    expected = [[1, 2], [3, 4]]

    assert to_python_rows([[1, 2], [3, 4]]) == expected
    assert to_python_rows(((1, 2), (3, 4))) == expected
    assert to_python_rows(np.array(expected, dtype=np.int32)) == expected
    assert to_python_rows(np.array(expected, dtype=np.uint8)) == expected
    assert to_python_rows(np.array(expected, dtype=object)) == expected
    assert to_python_rows(np.array([[10**30, 2**63]], dtype=object)) == [[10**30, 2**63]]
    assert to_python_rows(np.array([[3.0, 4.0]])) == [[3, 4]]
    assert to_python_rows([]) == []
    assert to_python_rows(np.zeros((0, 3), dtype=np.int64)) == []

    mixed = np.empty((1, 2), dtype=object)
    mixed[0, 0] = np.int64(5)
    mixed[0, 1] = 10**25
    assert to_python_rows(mixed) == [[5, 10**25]]


@pytest.mark.parametrize(
    "invalid",
    [
        [1, 2, 3],
        np.zeros((2, 2, 2), dtype=np.int64),
        [[1, 2], [3]],
        [[1.5, 2.0]],
        [[1, 2], [3, None]],
        [["a", "b"]],
        5,
    ],
)
def test_to_python_rows_rejects_non_integer_input(invalid):
    with pytest.raises(ValueError):
        to_python_rows(invalid)


# The canonical form is a function of the lattice alone, so every check below compares two
# descriptions of the *same* lattice.  The cases cover the shapes the orbit algebra
# produces: rank 0, rank 1, full rank, hand-built saturated kernels, and the primitive
# rational-nullspace basis of ``[[2, 1, 1]]``, which is not saturated and therefore spans a
# properly smaller lattice than the saturated kernel of the same matrix.

_CANONICAL_CASES = {
    "rank_zero": np.zeros((3, 0), dtype=np.int64),
    "rank_one_third_axis": np.array([[0], [0], [6]], dtype=np.int64),
    "rank_one_primitive": np.array([[3], [4], [0]], dtype=np.int64),
    "full_rank_identity": np.eye(3, dtype=np.int64),
    "full_rank_unimodular": np.array([[2, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.int64),
    "unsaturated_rational_kernel": np.array([[-1, -1], [2, 0], [0, 2]], dtype=np.int64),
    "saturated_primitive_kernel": np.array([[1, 0], [0, 1], [-2, -1]], dtype=np.int64),
    "saturated_kernel_of_two_constraints": saturated_kernel([[2, 1, 1, 0], [0, 1, -1, 3]]),
}


def _unimodular(size: int, rng) -> np.ndarray:
    """A random unimodular ``size`` x ``size`` matrix, i.e. an exact change of lattice basis."""
    matrix = np.eye(size, dtype=np.int64)
    for _ in range(4 * size + 4):
        left, right = (int(value) for value in rng.integers(0, size, 2))
        if left == right:
            continue
        matrix[:, left] += int(rng.choice([-3, -2, -1, 1, 2, 3])) * matrix[:, right]
    assert abs(int(Matrix(matrix.tolist()).det())) == 1
    return matrix


def _lattice_coordinates(basis: np.ndarray, vector, bound: int = _COEFFICIENT_BOUND):
    """Exact bounded search for integer coordinates of ``vector`` in ``basis`` columns.

    Independent of the module under test: it enumerates integer coefficient vectors and
    multiplies them out in Python integers.  ``None`` means "no representation with
    coefficients of magnitude at most ``bound``", which for the small cases used here is the
    same as "not in the lattice".
    """
    rows = basis.tolist()
    for coefficients in itertools.product(range(-bound, bound + 1), repeat=basis.shape[1]):
        if all(
            sum(entry * coefficient for entry, coefficient in zip(row, coefficients))
            == vector[position]
            for position, row in enumerate(rows)
        ):
            return list(coefficients)
    return None


def _assert_canonical_structure(canonical: np.ndarray) -> None:
    """Check the documented column structure of a canonical basis."""
    rows = canonical.tolist()
    previous_pivot = -1
    for column in range(canonical.shape[1]):
        nonzero = [row for row in range(canonical.shape[0]) if rows[row][column] != 0]
        assert nonzero, "a canonical column is zero"
        pivot = nonzero[-1]
        assert rows[pivot][column] > 0, "the pivot of a column must be positive"
        assert pivot > previous_pivot, "pivot rows must increase from column to column"
        assert all(rows[pivot][left] == 0 for left in range(column)), "pivot row not cleared"
        previous_pivot = pivot


@pytest.mark.parametrize("name", sorted(_CANONICAL_CASES))
def test_canonical_lattice_basis_depends_on_the_lattice_alone(name):
    basis = np.asarray(_CANONICAL_CASES[name], dtype=np.int64)
    rng = np.random.default_rng(SEED + len(name))
    columns = int(basis.shape[1])
    canonical = canonical_lattice_basis(basis)

    assert canonical.dtype == np.int64
    assert canonical.shape == (basis.shape[0], _oracle_rank(basis))
    _assert_canonical_structure(canonical)

    variants = [basis, np.hstack([basis, basis]), np.hstack([basis, -basis])]
    if columns:
        variants.append(basis @ _unimodular(columns, rng))
        variants.append(basis[:, rng.permutation(columns)])
        variants.append(np.hstack([basis, basis @ _unimodular(columns, rng)]))
    for variant in variants:
        np.testing.assert_array_equal(canonical_lattice_basis(variant), canonical)


@pytest.mark.parametrize("name", sorted(_CANONICAL_CASES))
def test_canonical_lattice_basis_spans_the_input_lattice_both_ways(name):
    basis = np.asarray(_CANONICAL_CASES[name], dtype=np.int64)
    canonical = canonical_lattice_basis(basis)

    for column in range(canonical.shape[1]):
        vector = [int(value) for value in canonical[:, column]]
        assert _lattice_coordinates(basis, vector) is not None
    for column in range(basis.shape[1]):
        vector = [int(value) for value in basis[:, column]]
        assert _lattice_coordinates(canonical, vector) is not None
    assert int(Matrix(basis.tolist()).rank()) == int(Matrix(canonical.tolist()).rank())


def test_canonical_lattice_basis_preserves_the_lattice_without_saturating_it():
    # The primitive rational-nullspace columns of [[2, 1, 1]] miss (1, 1, -3), so they span
    # a sublattice of the saturated kernel: canonicalizing must not repair that.
    unsaturated = np.array([[-1, -1], [2, 0], [0, 2]], dtype=np.int64)
    saturated = canonical_lattice_basis(saturated_kernel([[2, 1, 1]]))

    canonical = canonical_lattice_basis(unsaturated)

    assert _lattice_coordinates(canonical, (1, 1, -3)) is None
    assert _lattice_coordinates(saturated, (1, 1, -3)) is not None
    assert not np.array_equal(canonical, saturated)
    assert _lattice_coordinates(canonical, (1, 1, -3)) is None
    for column in range(canonical.shape[1]):
        vector = [int(value) for value in canonical[:, column]]
        assert _lattice_coordinates(unsaturated, vector) is not None


_CANONICAL_FORM_CASES = {
    "zero_columns": (np.zeros((3, 0), dtype=np.int64), np.zeros((3, 0), dtype=np.int64)),
    "empty": (np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.int64)),
    "no_rows": (np.zeros((0, 3), dtype=np.int64), np.zeros((0, 0), dtype=np.int64)),
    "all_zero": (np.zeros((3, 2), dtype=np.int64), np.zeros((3, 0), dtype=np.int64)),
    "already_canonical": (
        np.array([[4], [2], [0]], dtype=np.int64),
        np.array([[4], [2], [0]], dtype=np.int64),
    ),
    "negative_pivot": (
        np.array([[0], [-5]], dtype=np.int64),
        np.array([[0], [5]], dtype=np.int64),
    ),
    "primitive_single_row": (
        np.array([[2, 1, 1]], dtype=np.int64),
        np.array([[1]], dtype=np.int64),
    ),
    "dependent_columns": (
        np.array([[1, 2], [2, 4]], dtype=np.int64),
        np.array([[1], [2]], dtype=np.int64),
    ),
    "both_columns_kept": (
        np.array([[2, 4], [0, 2]], dtype=np.int64),
        np.array([[2, 0], [0, 2]], dtype=np.int64),
    ),
    "sheared_triangle": (
        np.array([[3, 0], [2, 5]], dtype=np.int64),
        np.array([[15, 9], [0, 1]], dtype=np.int64),
    ),
}


@pytest.mark.parametrize("name", sorted(_CANONICAL_FORM_CASES))
def test_canonical_lattice_basis_documented_conventions(name):
    basis, expected = _CANONICAL_FORM_CASES[name]

    canonical = canonical_lattice_basis(basis)

    assert canonical.dtype == np.int64
    np.testing.assert_array_equal(canonical, expected)
    _assert_canonical_structure(canonical)
    # Canonicalizing is idempotent, which is what "canonical" means operationally.
    np.testing.assert_array_equal(canonical_lattice_basis(canonical), canonical)


def test_canonical_lattice_basis_raises_for_entries_that_do_not_fit_int64():
    # The canonical form is exact, so it has to *refuse* to enter the int64 tensor layer.
    with pytest.raises(IntegerRangeError):
        canonical_lattice_basis([[2**70]])
    with pytest.raises(IntegerRangeError):
        canonical_lattice_basis(np.array([[2**63], [1]], dtype=object))
    with pytest.raises(IntegerRangeError):
        canonical_lattice_basis(np.array([[10**25, 1], [0, 3]], dtype=object))

    # The boundary is exactly int64, not a margin below it.
    np.testing.assert_array_equal(
        canonical_lattice_basis(np.array([[2**62]], dtype=object)),
        np.array([[2**62]], dtype=np.int64),
    )


def test_invariant_kernel_returns_canonical_columns():
    # A 2-atom cluster with one label, so the label basis is symmetric under the atom swap,
    # plus the lattice-frame mirror diag(1, 1, -1) as its only stabilizer constraint.
    label = label_symmetric_basis([0, 0])
    mirror = TensorAction(
        rotation=np.eye(3),
        permutation=(0, 1),
        order=2,
        scaled_rotation=np.diag(np.array([1, 1, -1], dtype=np.int64)),
    )

    basis, dimension = invariant_kernel(label, [mirror], order=2)

    assert basis.dtype == np.int64
    # Six label-symmetric classes survive the mirror only if they carry no z axis, so the
    # invariant dimension is four and the canonical basis selects exactly those columns.
    assert dimension == 4
    assert basis.shape == (label.shape[1], 4)
    # Every column is invariant under the stabilizer once expanded into component space,
    # and each one is a label-class indicator, so it is a column of the label basis.
    exact = label @ basis
    np.testing.assert_array_equal(mirror.apply_scaled_columns(exact), exact)
    label_columns = {tuple(int(value) for value in column) for column in label.T}
    assert {tuple(int(value) for value in column) for column in exact.T} <= label_columns
    np.testing.assert_array_equal(basis, canonical_lattice_basis(basis))
    # The stored columns are canonical, so an exact change of parameter basis is invisible.
    rng = np.random.default_rng(SEED + 7)
    np.testing.assert_array_equal(
        canonical_lattice_basis(basis @ _unimodular(dimension, rng)), basis
    )


@pytest.mark.parametrize("corruption", ["scaled_column", "dropped_column"])
def test_canonical_lattice_basis_rejects_a_result_that_changes_the_lattice(monkeypatch, corruption):
    # The lattice check inside the function is the only thing standing between a wrong
    # Hermite call and a parameterization that silently fits the wrong subspace, so it is
    # exercised with a deliberately corrupted normal form.
    from sympy.matrices import normalforms

    honest = normalforms.hermite_normal_form

    def corrupted(matrix):
        result = honest(matrix)
        if corruption == "scaled_column":
            # A basis vector doubled spans a proper sublattice of the input lattice.
            result[:, 0] = 2 * result[:, 0]
        else:
            # A dropped basis vector leaves a rank-deficient result.
            result = result[:, 1:]
        return result

    monkeypatch.setattr(normalforms, "hermite_normal_form", corrupted)

    with pytest.raises(RuntimeError, match="integer combination"):
        canonical_lattice_basis([[2, 1], [1, 3]])
