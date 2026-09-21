"""Exact rank certification and saturated integer kernels for lattice orbit algebra.

Integer matrices are interpreted the usual way: the kernel of ``matrix`` is
``{x in Z^n : matrix @ x == 0}`` with ``matrix @ x`` evaluated over the
integers, so a kernel basis is meaningful as a lattice basis.

Rank over the rationals is certified modularly rather than by exact
division-free elimination on the full integers.  Reducing modulo a prime is a
ring homomorphism, so the rank of the reduction never exceeds the rank over the
rationals: every modular rank is a *lower* bound, and a modular rank that
already attains ``min(shape)`` settles the question in one pass.  Otherwise the
accumulating product of *distinct* primes supplies the matching upper bound --
were the exact rank larger than every modular rank seen so far, the nonzero
minor that witnesses the larger rank would be divisible by each prime drawn, so
its absolute value would be at least their product.  A Hadamard bound on that
minor (the product of the ``size`` largest integer row norms) therefore
certifies the running maximum as soon as the accumulated prime product exceeds
it.

Primes come from :func:`prime_stream`, an on-demand stream: it starts with
:data:`RANK_PRIMES` and then takes successive ``prevprime`` steps.  It is not an
infinite prime enumeration -- once ``prevprime`` has nothing smaller to offer
the stream ends, and :func:`certified_rank` reports
:class:`~mlfcs.exceptions.RankCertificateError` if the certificate was never
reached.  The result is nevertheless *exact* for integer inputs of arbitrary
magnitude: entries are reduced modulo a prime with Python integer remainder
before they ever reach a fixed-width buffer, so ``10**30`` entries are ranked
without truncation, and only the elimination loop itself (where residues are
smaller than ``2**31``, hence products smaller than ``2**62``) runs in int64.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterator

import numpy as np

from mlfcs.exceptions import IntegerRangeError, RankCertificateError

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

#: Fast-path pair of primes just below 2**31; products of two residues stay in int64.
RANK_PRIMES: tuple[int, int] = (2147483647, 2147483629)


def to_python_rows(matrix) -> list[list[int]]:
    """Rows of ``matrix`` as Python ints, exactly.

    Nested lists or tuples, NumPy integer arrays of any width and object arrays
    holding Python ints (arbitrary precision) are accepted.  Integral floats are
    converted exactly; any other entry, and input that is not two-dimensional,
    raises ``ValueError``.  The empty sequence reads as the 0x0 matrix.
    """
    return _integer_rows(matrix)[2]


def _integer_rows(matrix) -> tuple[int, int, list[list[int]]]:
    """Return ``(height, width, rows)`` of exact Python-int rows for any input."""
    array = matrix if isinstance(matrix, np.ndarray) else np.asarray(matrix, dtype=object)
    if array.ndim == 1 and array.size == 0:
        return 0, 0, []
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D integer matrix, got shape {array.shape}")
    height, width = int(array.shape[0]), int(array.shape[1])
    if array.dtype.kind in "iub":
        return height, width, [[int(value) for value in row] for row in array.tolist()]
    rows = [
        [_exact_int(value, position=(index, column)) for column, value in enumerate(values)]
        for index, values in enumerate(array.tolist())
    ]
    return height, width, rows


def _exact_int(value, position: tuple[int, int]) -> int:
    """Convert one input entry to a Python int or reject it."""
    try:
        return operator.index(value)
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        if math.isfinite(as_float) and as_float.is_integer():
            return int(as_float)
    raise ValueError(f"matrix entry at {position} is not an integer: {value!r}")


def _row_residues(rows: list[list[int]], width: int, prime: int) -> np.ndarray:
    """Reduce exact Python-int rows into a fresh int64 residue buffer."""
    residues = np.zeros((len(rows), width), dtype=np.int64)
    for index, row in enumerate(rows):
        residues[index] = [value % prime for value in row]
    return residues


def _echelon(residues: np.ndarray, prime: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-reduce a residue buffer mod ``prime`` in place; see :func:`modular_echelon`."""
    echelon = residues
    height, width = echelon.shape
    order = list(range(height))
    pivot_columns: list[int] = []
    pivot_rows: list[int] = []
    row = 0
    for column in range(width):
        if row >= height:
            break
        candidates = np.nonzero(echelon[row:, column])[0]
        if candidates.size == 0:
            continue
        picked = row + int(candidates[0])
        if picked != row:
            echelon[[row, picked]] = echelon[[picked, row]]
            order[row], order[picked] = order[picked], order[row]
        pivot_index = row
        inverse = pow(int(echelon[pivot_index, column]), prime - 2, prime)
        echelon[pivot_index, column:] = (echelon[pivot_index, column:] * inverse) % prime
        pivot_columns.append(column)
        pivot_rows.append(order[pivot_index])
        row += 1
        block = echelon[row:, :]
        factors = block[:, column].copy()
        nonzero = factors != 0
        if nonzero.any():
            # Both operands are residues below 2**31, so the products stay in int64.
            block[nonzero] = (
                block[nonzero] - np.outer(factors[nonzero], echelon[pivot_index])
            ) % prime
    return echelon, np.array(pivot_columns, dtype=np.int64), np.array(pivot_rows, dtype=np.int64)


def modular_echelon(matrix, prime: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row echelon form of ``matrix`` modulo ``prime``.

    Returns the echelon rows as an int64 array shaped like the input, the pivot
    columns, and the pivot rows referencing the *input* row order (survivors of
    the row swaps).  Entries are reduced modulo ``prime`` with Python integer
    remainder first, so arbitrary-precision input is accepted; the elimination
    loop itself stays in int64, where products of two residues below ``2**31``
    cannot overflow.
    """
    _height, width, rows = _integer_rows(matrix)
    if prime < 2:
        raise ValueError(f"prime must be at least 2, got {prime}")
    return _echelon(_row_residues(rows, width, prime), prime)


def modular_rank(matrix, prime: int) -> int:
    """Rank of ``matrix`` modulo ``prime``; never exceeds the exact rank."""
    return int(modular_echelon(matrix, prime)[1].size)


def hadamard_bound(matrix, size: int) -> int:
    """Exact upper bound on the absolute value of any ``size`` x ``size`` minor.

    The bound multiplies the ``size`` largest row norms, each one
    ``isqrt(sum of squares) + 1`` in exact Python integer arithmetic (never
    float), which strictly exceeds the Hadamard product of Euclidean row norms
    and therefore the minor itself.  ``size <= 0`` returns 1; a matrix with
    fewer than ``size`` rows has only zero ``size``-minors, so multiplying the
    available row norms stays a valid bound.
    """
    if size <= 0:
        return 1
    return _hadamard_bound(_integer_rows(matrix)[2], size)


def _hadamard_bound(rows: list[list[int]], size: int) -> int:
    norms = sorted(
        (math.isqrt(sum(value * value for value in row)) + 1 for row in rows),
        reverse=True,
    )
    bound = 1
    for norm in norms[:size]:
        bound *= norm
    return bound


def prime_stream() -> Iterator[int]:
    """On-demand stream of distinct primes below ``2**31``, descending.

    The first two entries are exactly :data:`RANK_PRIMES`; the rest are the
    successive ``prevprime`` values (imported lazily so the fast path needs no
    ``sympy``).  This is an on-demand stream rather than a prime enumeration:
    once ``prevprime`` has no smaller prime to offer, the stream stops.
    """
    yield from RANK_PRIMES
    from sympy.ntheory.generate import prevprime

    candidate = RANK_PRIMES[1]
    while True:
        try:
            smaller = prevprime(candidate)
        except ValueError:  # sympy has no preceding prime left to offer
            return
        if smaller is None or smaller >= candidate:
            return
        candidate = int(smaller)
        yield candidate


def certified_pivots(matrix) -> tuple[int, np.ndarray, np.ndarray]:
    """Exact rank plus a maximal independent row set and column set.

    Returns ``(rank, pivot_rows, pivot_columns)`` with indices into the input
    rows and columns.  Both sets come from the modular echelon of a prime that
    attained the returned rank, so the selected ``rank`` x ``rank`` submatrix
    has a determinant that is nonzero modulo that prime and hence nonzero over
    the rationals.  The selected rows are thus independent, and ``rank``
    independent rows of a rank-``rank`` row space span it: ``matrix[pivot_rows]``
    has exactly the same kernel as ``matrix``, over both the rationals and the
    integers.
    """
    height, width, rows = _integer_rows(matrix)
    target = min(height, width)
    if target == 0:
        empty = np.empty(0, dtype=np.int64)
        return 0, empty, empty
    bound = _hadamard_bound(rows, target)
    best = 0
    best_rows = np.empty(0, dtype=np.int64)
    best_columns = np.empty(0, dtype=np.int64)
    prime_product = 1
    for prime in prime_stream():
        pivots = _echelon(_row_residues(rows, width, prime), prime)
        columns, pivot_rows = pivots[1], pivots[2]
        if columns.size > best:
            best = int(columns.size)
            best_rows = pivot_rows
            best_columns = columns
        if best == target:
            break
        prime_product *= prime
        if prime_product > bound:
            break
    else:
        raise RankCertificateError(
            f"rank of a {height}x{width} integer matrix is not certified: the modular "
            f"lower bound {best} stays below {target} and the prime stream ended before "
            f"the Hadamard bound {bound} was exceeded"
        )
    return best, best_rows, best_columns


def certified_rank(matrix) -> int:
    """Exact rank over the rationals, certified by accumulated distinct primes.

    See the module docstring for the certificate.  ``int32``/``int64``/object
    and Python-int inputs, including mixed magnitudes, are ranked exactly and
    without overflow or truncation.  Empty shapes and zero-column shapes have
    rank 0.
    """
    return certified_pivots(matrix)[0]


def saturated_kernel(matrix) -> np.ndarray:
    """Exact, saturated basis of ``{x in Z^n : matrix @ x == 0}``.

    The basis is read off the Smith normal form decomposition ``S = U @ A @ V``
    with ``U`` and ``V`` unimodular: ``V`` is a ``Z``-basis of ``Z^n``, so its
    columns at zero diagonal positions of ``S`` form a basis of the integer
    kernel itself and are saturated by construction.  Primitive
    rational-nullspace columns are not: for ``[[2, 1, 1]]`` the primitive
    directions ``(-1, 2, 0)`` and ``(-1, 0, 2)`` generate only a sublattice and
    miss the integer kernel vector ``(1, 1, -3)``.  This construction reaches
    ``(1, 1, 1)`` in the kernel of ``[[1, -2, 1]]``.

    A 0-row matrix has the full space as kernel (the identity of size ``n``); a
    matrix with no columns yields shape ``(0, 0)``.  Raises
    :class:`~mlfcs.exceptions.IntegerRangeError` when a basis entry does not fit
    int64.
    """
    height, width, rows = _integer_rows(matrix)
    if width == 0:
        return np.zeros((0, 0), dtype=np.int64)
    if height == 0:
        return np.eye(width, dtype=np.int64)
    from sympy import ZZ, Matrix
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.normalforms import smith_normal_decomp

    domain = DomainMatrix.from_Matrix(Matrix(rows)).convert_to(ZZ)
    diagonal, _left, right = smith_normal_decomp(domain)
    diagonal_matrix = diagonal.to_Matrix()
    free = [
        column
        for column in range(width)
        if all(diagonal_matrix[row, column] == 0 for row in range(height))
    ]
    right_matrix = right.to_Matrix()
    basis = np.zeros((width, len(free)), dtype=np.int64)
    for position, column in enumerate(free):
        for row in range(width):
            value = int(right_matrix[row, column])
            if not _INT64_MIN <= value <= _INT64_MAX:
                raise IntegerRangeError(
                    f"kernel basis entry {value} at ({row}, {position}) does not fit int64"
                )
            basis[row, position] = value
    return basis


def canonical_lattice_basis(basis) -> np.ndarray:
    r"""Return the canonical column basis of the integer lattice spanned by ``basis``.

    A saturated kernel from Smith normal form spans the right lattice but its columns are
    not unique for that lattice, so two implementations (or two sympy versions) may return
    different columns for the same subspace. The column Hermite normal form is unique for a
    given lattice, so applying it makes the result comparable bit by bit.

    ``basis`` is an integer matrix whose *columns* span the lattice; the result is the
    column Hermite normal form of that matrix, computed by
    ``sympy.matrices.normalforms.hermite_normal_form`` (deferred import) on those columns.
    The convention that comes back -- pinned by the tests, because it is the caller's only
    handle on the result -- is:

    * the shape is ``(height, rank)`` where ``height`` is the input's row count: dependent
      columns are dropped, so a rank-deficient input returns fewer columns than it was
      given. A basis with no columns, and an all-zero basis, both return ``(height, 0)``,
      and an input with no rows returns ``(0, 0)``: the zero lattice is reported as zero
      columns in the ambient space of the input;
    * the columns are a ``Z``-basis of exactly the same lattice (checked below), ordered by
      increasing pivot row;
    * the *pivot* of a column is its bottom-most nonzero entry: it is positive, pivot rows
      strictly increase from column to column, and the pivot row of column ``j`` is zero to
      the left of ``j``. Entries above a pivot are reduced modulo it for the rows sympy's
      algorithm processes, but the top rows it never reaches keep whatever elimination left
      there, so no global reduction bound may be assumed -- a future reader must not read a
      triangular normal form into the array;
    * every entry is exact; entries that do not fit ``int64`` raise
      :class:`~mlfcs.exceptions.IntegerRangeError`. That is the documented boundary of the
      module: the ranking/kernel layer is arbitrary-precision, the tensor layer is int64.

    Raises ``RuntimeError`` if the result does not span the same lattice as the input
    (checked exactly, in both directions, in Python integers).

    Examples, all pinned by the tests: ``[[4], [2], [0]]`` is already canonical,
    ``[[3, 0], [2, 5]]`` becomes ``[[15, 9], [0, 1]]``, and the two primitive columns
    ``(-1, 2, 0)``/``(-1, 0, 2)``, which span only a sublattice of the integer kernel of
    ``[[2, 1, 1]]``, canonicalize to columns of *that sublattice*: canonicalizing a lattice
    never saturates it, it only makes the basis unique.
    """
    height, width, rows = _integer_rows(basis)
    if width == 0 or height == 0:
        return np.zeros((height, 0), dtype=np.int64)
    from sympy import Matrix
    from sympy.matrices.normalforms import hermite_normal_form

    table = hermite_normal_form(Matrix(rows)).tolist()
    canonical = np.zeros((height, len(table[0])), dtype=np.int64)
    for row, values in enumerate(table):
        for column, value in enumerate(values):
            entry = int(value)
            if not _INT64_MIN <= entry <= _INT64_MAX:
                raise IntegerRangeError(
                    f"canonical lattice basis entry {entry} at ({row}, {column}) does not "
                    "fit int64; reduce the primitive cell before building orbit algebra"
                )
            canonical[row, column] = entry
    _verify_same_lattice(rows, height, canonical)
    return canonical


def _column_lattice_test(rows: list[list[int]], height: int) -> Callable[[list[int]], bool]:
    """Return an exact membership test for the lattice spanned by the columns of ``rows``.

    The fast path covers every basis this module is used on: when the columns are
    independent the coordinates of a vector are unique, so an exact rational solve decides
    membership as soon as the diagonal denominators are inspected.  With dependent columns
    a rational solution can be non-integral while an integer one exists, so the decision
    falls back to the Smith normal form criterion of :func:`_lattice_oracle`, which is exact
    for any integer matrix but much more expensive.
    """
    from sympy import Matrix

    matrix = Matrix(rows)
    smith: tuple[list[int], list[list[int]]] | None = None

    def contains(vector: list[int]) -> bool:
        nonlocal smith
        try:
            solution, parameters = matrix.gauss_jordan_solve(Matrix(vector))
        except ValueError:  # the columns cannot reach the vector at all
            return False
        if not len(parameters):
            return all(value.q == 1 for value in solution)
        if smith is None:
            smith = _lattice_oracle(rows, height)
        return _in_oracle_lattice(smith[0], smith[1], vector)

    return contains


def _lattice_oracle(rows: list[list[int]], height: int) -> tuple[list[int], list[list[int]]]:
    """Return ``(divisors, left)`` that decide exact membership of the column lattice.

    ``smith_normal_decomp`` returns ``(D, U, V)`` with ``matrix == U**-1 D V**-1`` (pinned
    by the tests), so the columns of ``matrix`` are ``{U**-1 D y}`` and a vector ``v`` is an
    integer combination of them exactly when ``U @ v`` is divisible by the diagonal of ``D``
    in the leading positions and zero beyond them.
    """
    from sympy import ZZ, Matrix
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.normalforms import smith_normal_decomp

    domain = DomainMatrix.from_Matrix(Matrix(rows)).convert_to(ZZ)
    diagonal, left, _right = smith_normal_decomp(domain)
    diagonal_matrix = diagonal.to_Matrix()
    left_rows = [[int(value) for value in row] for row in left.to_Matrix().tolist()]
    width = len(rows[0])
    divisors = [int(diagonal_matrix[index, index]) for index in range(min(height, width))]
    return divisors, left_rows


def _in_oracle_lattice(divisors: list[int], left: list[list[int]], column: list[int]) -> bool:
    """Exact test that ``column`` is an integer combination of an oracle's columns."""
    for index, row in enumerate(left):
        value = sum(entry * component for entry, component in zip(row, column, strict=True))
        if index >= len(divisors) or divisors[index] == 0:
            if value:
                return False
        elif value % divisors[index]:
            return False
    return True


def _verify_same_lattice(rows: list[list[int]], height: int, canonical: np.ndarray) -> None:
    """Raise ``RuntimeError`` unless the canonical columns span the input lattice.

    Both directions are checked, because either failure mode is silent otherwise: an
    enlarged result would fit parameters the constraints do not allow, a shrunk one would
    drop invariant parameters.
    """
    width = len(rows[0])
    canonical_rows = [[int(value) for value in row] for row in canonical.tolist()]
    in_input = _column_lattice_test(rows, height)
    for column in range(canonical.shape[1]):
        if not in_input([row[column] for row in canonical_rows]):
            raise RuntimeError(
                f"canonical lattice basis column {column} is not an integer combination of "
                f"the {width} input columns"
            )
    in_canonical = _column_lattice_test(canonical_rows, height)
    for column in range(width):
        if not in_canonical([row[column] for row in rows]):
            raise RuntimeError(
                f"input column {column} is not an integer combination of the "
                f"{canonical.shape[1]} canonical lattice basis columns"
            )


def verify_kernel(matrix, basis) -> None:
    """Exact check that ``matrix @ basis`` is the zero matrix over the integers.

    Entries are multiplied as Python ints, so the check is exact at any
    magnitude.  Raises ``ValueError`` naming the failing row, column and
    residual; empty matrices are valid input.
    """
    height, width, rows = _integer_rows(matrix)
    basis_height, basis_width, basis_rows = _integer_rows(basis)
    if basis_height != width:
        raise ValueError(f"kernel basis has {basis_height} rows but the matrix has {width} columns")
    for row in range(height):
        for column in range(basis_width):
            residual = sum(rows[row][index] * basis_rows[index][column] for index in range(width))
            if residual:
                raise ValueError(
                    f"matrix @ basis is nonzero at row {row}, column {column}: residual {residual}"
                )


__all__ = [
    "RANK_PRIMES",
    "canonical_lattice_basis",
    "certified_pivots",
    "certified_rank",
    "hadamard_bound",
    "modular_echelon",
    "modular_rank",
    "prime_stream",
    "saturated_kernel",
    "to_python_rows",
    "verify_kernel",
]
