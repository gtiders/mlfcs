"""Shared exact integer-matrix utilities for MLFCS numerical algorithms."""

from __future__ import annotations

import operator

import numpy as np
from numba import njit

from mlfcs.foundation.arrays import as_int64_array
from mlfcs.foundation.errors import ArithmeticRangeError

RANK_PRIMES = (2147483647, 2147483629)


def _is_prime(candidate: int) -> bool:
    """Return whether ``candidate`` is prime using Miller-Rabin bases that are
    deterministic below the 32-bit limit."""
    if candidate < 2:
        return False
    for prime in (2, 3, 5, 7, 11):
        if candidate % prime == 0:
            return candidate == prime
    odd_part, power_of_two = candidate - 1, 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power_of_two += 1
    for base in (2, 3, 5, 7, 11):
        residue = pow(base, odd_part, candidate)
        if residue in (1, candidate - 1):
            continue
        for _ in range(power_of_two - 1):
            residue = residue * residue % candidate
            if residue == candidate - 1:
                break
        else:
            return False
    return True


def prime_stream():
    """Yield descending primes for modular rank and exact-certification routines.

    All returned primes are below ``2**31`` so products of reduced residues
    remain safely representable in signed int64 arithmetic.
    """
    yield from RANK_PRIMES
    candidate = RANK_PRIMES[-1] - 2
    while candidate >= 3:
        if _is_prime(candidate):
            yield candidate
        candidate -= 2
    yield 2


def to_python_rows(matrix: object) -> list[list[int]]:
    """Convert a two-dimensional integer matrix to unbounded Python integers."""
    array = matrix if isinstance(matrix, np.ndarray) else np.asarray(matrix, dtype=object)
    if array.ndim == 1 and array.size == 0:
        return []
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D integer matrix, got shape {array.shape}")
    if array.dtype.kind in "iub":
        return [[int(value) for value in row] for row in array.tolist()]
    rows = []
    for row_index, row in enumerate(array.tolist()):
        values = []
        for column_index, value in enumerate(row):
            try:
                values.append(operator.index(value))
            except TypeError as error:
                raise ValueError(
                    f"matrix entry at {(row_index, column_index)} is not an integer: {value!r}"
                ) from error
        rows.append(values)
    return rows


def as_int64(values: object, *, context: str) -> np.ndarray:
    """Validate integer input for exact operations restricted to int64 arithmetic.

    Unlike general array normalization, this helper accepts only explicit
    integer dtypes and translates range failures into ``ArithmeticRangeError``.
    Values use the symmetric int64 domain excluding ``INT64_MIN``.
    """
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{context} must contain integers")
    try:
        return as_int64_array(array, name=context)
    except OverflowError as error:
        raise ArithmeticRangeError(f"{context} does not fit the symmetric int64 domain") from error


def as_int64_matrix(matrix: object, *, name: str = "matrix") -> np.ndarray:
    """Return a readonly C-contiguous int64 matrix.

    A one-dimensional empty input is interpreted as a ``(0, 0)`` matrix.
    Other nonmatrix inputs are rejected. Range failures raise
    ``ArithmeticRangeError``, consistent with :func:`as_int64`.
    """
    array = np.asarray(matrix)
    if array.ndim == 1 and array.size == 0:
        array = np.empty((0, 0), dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("expected a 2-D integer matrix")
    try:
        return as_int64_array(array, name=name)
    except OverflowError as error:
        raise ArithmeticRangeError(f"{name} does not fit the symmetric int64 domain") from error


def determinant_3x3(matrix: object) -> int:
    """Return the exact determinant of a ``3 x 3`` integer matrix.

    The determinant is evaluated with Python integers, so the result is not
    restricted to int64 even though the input matrix is.
    """
    values = as_int64(matrix, context="matrix")
    if values.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    a, b, c, d, e, f, g, h, i = (int(value) for value in values.ravel())
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def adjugate_3x3(matrix: object) -> np.ndarray:
    """Return the exact integer adjugate of a ``3 x 3`` matrix.

    For matrix ``S``,

        S @ adj(S) = det(S) * I.

    The cofactors must remain representable in the supported int64 domain.
    """
    values = as_int64(matrix, context="matrix")
    if values.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    a, b, c, d, e, f, g, h, i = (int(value) for value in values.ravel())
    result = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    try:
        return as_int64_array(np.asarray(result, dtype=object), name="matrix adjugate")
    except OverflowError as error:
        raise ArithmeticRangeError("matrix adjugate does not fit in int64") from error


@njit(cache=True)
def modular_power(base, exponent, modulus):
    """Return ``base**exponent mod modulus`` by repeated squaring.

    Inputs are chosen so every reduced multiplication fits signed int64.
    """
    result = 1
    while exponent:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


@njit(cache=True)
def echelon(matrix, prime):
    """Compute forward row-echelon form over the finite field ``F_prime``.

    Gaussian elimination is performed modulo the prime ``prime``. The function
    returns the reduced working matrix together with the pivot-column indices
    and the original row indices chosen as pivots.

    Parameters
    ----------
    matrix
        Integer matrix of shape ``(m, n)``.
    prime
        Prime modulus below ``2**31``.

    Returns
    -------
    echelon
        Matrix in forward modular row-echelon form.
    pivot_columns
        Column indices of the pivots.
    pivot_rows
        Original input-row indices corresponding to those pivots.
    """
    a = matrix % prime
    height, width = a.shape
    order = np.arange(height, dtype=np.int64)
    columns = np.empty(min(height, width), dtype=np.int64)
    rows = np.empty(min(height, width), dtype=np.int64)
    rank = 0
    for column in range(width):
        if rank == height:
            break
        picked = -1
        for i in range(rank, height):
            if a[i, column]:
                picked = i
                break
        if picked < 0:
            continue
        for j in range(width):
            a[rank, j], a[picked, j] = a[picked, j], a[rank, j]
        order[rank], order[picked] = order[picked], order[rank]
        inverse = modular_power(a[rank, column], prime - 2, prime)
        for j in range(column, width):
            a[rank, j] = a[rank, j] * inverse % prime
        columns[rank], rows[rank] = column, order[rank]
        for i in range(rank + 1, height):
            factor = a[i, column]
            for j in range(column, width):
                a[i, j] = (a[i, j] - factor * a[rank, j]) % prime
        rank += 1
    return a, columns[:rank], rows[:rank]


__all__ = [
    "RANK_PRIMES",
    "adjugate_3x3",
    "as_int64",
    "as_int64_matrix",
    "determinant_3x3",
    "echelon",
    "modular_power",
    "prime_stream",
    "to_python_rows",
]
