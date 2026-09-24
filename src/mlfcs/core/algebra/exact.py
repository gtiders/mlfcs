"""Exact rank certification and saturated kernels for integer linear algebra.

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
:class:`~mlfcs.core.errors.RankCertificateError` if the certificate was never
reached.  The result is nevertheless *exact* for integer inputs of arbitrary
magnitude: entries are reduced modulo a prime with Python integer remainder
before they ever reach a fixed-width buffer, so ``10**30`` entries are ranked
without truncation, and only the elimination loop itself (where residues are
smaller than ``2**31``, hence products smaller than ``2**62``) runs in int64.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator

import numpy as np
from numba import njit

from mlfcs.core.errors import RankCertificateError

#: Fast-path pair of primes just below 2**31; products of two residues stay in int64.
RANK_PRIMES: tuple[int, int] = (2147483647, 2147483629)


def to_python_rows(matrix) -> list[list[int]]:
    """Rows of ``matrix`` as Python ints, exactly.

    Nested lists or tuples, NumPy integer arrays of any width and object arrays
    holding Python ints (arbitrary precision) are accepted. Floating-point values
    are refused even when integral: an exact integer fact has to be declared as
    such. Any other entry, and input that is not two-dimensional, raises
    ``ValueError``. The empty sequence reads as the 0x0 matrix.
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
    raise ValueError(f"matrix entry at {position} is not an integer: {value!r}")


def _row_residues(rows: list[list[int]], width: int, prime: int) -> np.ndarray:
    """Reduce exact Python-int rows into a fresh int64 residue buffer."""
    residues = np.zeros((len(rows), width), dtype=np.int64)
    for index, row in enumerate(rows):
        residues[index] = [value % prime for value in row]
    return residues


@njit(cache=True)
def _modular_power(base: int, exponent: int, modulus: int) -> int:
    result = 1
    value = base % modulus
    while exponent:
        if exponent & 1:
            result = (result * value) % modulus
        value = (value * value) % modulus
        exponent >>= 1
    return result


@njit(cache=True)
def _echelon(residues: np.ndarray, prime: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-reduce a residue buffer mod ``prime`` in place; see :func:`modular_echelon`."""
    echelon = residues
    height, width = echelon.shape
    order = np.arange(height, dtype=np.int64)
    pivot_columns = np.empty(min(height, width), dtype=np.int64)
    pivot_rows = np.empty(min(height, width), dtype=np.int64)
    row = 0
    for column in range(width):
        if row >= height:
            break
        picked = -1
        for candidate in range(row, height):
            if echelon[candidate, column] != 0:
                picked = candidate
                break
        if picked < 0:
            continue
        if picked != row:
            for entry in range(width):
                temporary = echelon[row, entry]
                echelon[row, entry] = echelon[picked, entry]
                echelon[picked, entry] = temporary
            temporary_order = order[row]
            order[row] = order[picked]
            order[picked] = temporary_order
        inverse = _modular_power(int(echelon[row, column]), prime - 2, prime)
        for entry in range(column, width):
            echelon[row, entry] = (echelon[row, entry] * inverse) % prime
        pivot_columns[row] = column
        pivot_rows[row] = order[row]
        row += 1
        for target in range(row, height):
            factor = echelon[target, column]
            if factor == 0:
                continue
            for entry in range(column, width):
                # Residue products are below 2**62 and therefore exact in int64.
                echelon[target, entry] = (
                    echelon[target, entry] - factor * echelon[row - 1, entry]
                ) % prime
    return echelon, pivot_columns[:row], pivot_rows[:row]


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


def saturated_kernel(matrix, *, expected_nullity: int | None = None) -> np.ndarray:
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
    matrix with no columns yields shape ``(0, 0)``.  The returned object array
    contains Python ``int`` values, so the exact core has no fixed-width range
    boundary.  Consumers that need an int64 work buffer must check that boundary
    explicitly when they enter their compiled numerical kernel.

    The Smith decomposition is the saturation certificate.  This function also
    checks the two facts needed by consumers: the returned columns are
    annihilated exactly by ``matrix``, and their number is the rational nullity.
    By default that nullity is certified here.  A caller that already obtained
    it from :func:`certified_pivots` may pass ``expected_nullity`` to avoid
    repeating the same modular elimination.  It deliberately does not canonicalize the columns: a
    Hermite normal form changes only the coordinates used for the same lattice,
    adds no physical information, and is prohibitively expensive for the many
    small kernels built by a cluster space.
    """
    height, width, rows = _integer_rows(matrix)
    if width == 0:
        return np.zeros((0, 0), dtype=object)
    if height == 0:
        return np.eye(width, dtype=object)
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
    basis = np.zeros((width, len(free)), dtype=object)
    for position, column in enumerate(free):
        for row in range(width):
            basis[row, position] = int(right_matrix[row, column])
    verify_kernel(rows, basis)
    nullity = width - certified_rank(rows) if expected_nullity is None else int(expected_nullity)
    if not 0 <= nullity <= width:
        raise ValueError(f"expected_nullity must lie between 0 and {width}, got {nullity}")
    if basis.shape[1] != nullity:
        raise RuntimeError(
            "Smith kernel dimension does not match the certified nullity: "
            f"got {basis.shape[1]}, expected {nullity}"
        )
    return basis


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
