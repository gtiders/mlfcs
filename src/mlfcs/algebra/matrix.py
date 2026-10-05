"""Matrix products, lattice inverses and Euclidean arithmetic."""

from __future__ import annotations

import operator

import numpy as np
from numba import njit

from mlfcs._arrays import as_int64_array, require_allocation, require_bound
from mlfcs.errors import ArithmeticRangeError


def to_python_rows(matrix: object) -> list[list[int]]:
    """Convert a two-dimensional integer matrix to nested arbitrary-precision rows.

    Accept integer NumPy arrays, nested integer sequences, and object arrays
    whose entries implement the integer index protocol. Preserve row and column
    order. An empty sequence represents shape (0, 0); floats and nonmatrix
    inputs raise ValueError. No fixed-width range limit is applied.
    """
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
    """Prepare a readonly, C-contiguous int64 array for matrix operations.

    ``values`` is an array_like with an integer dtype and any shape; ``context``
    names it in diagnostics. Floating and object dtypes raise TypeError.
    Entries outside [-INT64_MAX, INT64_MAX] or excessive storage raise
    ArithmeticRangeError. Writable caller storage is copied.
    """
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{context} must contain integers")
    try:
        return as_int64_array(array, name=context)
    except OverflowError as error:
        raise ArithmeticRangeError(f"{context} does not fit the symmetric int64 domain") from error


def determinant_3x3(matrix: object) -> int:
    """Compute the determinant of a 3 by 3 matrix by cofactor expansion.

    ``matrix`` contains integers in the range accepted by as_int64. Return
    a Python int, which may exceed the input dtype's range. Invalid shape
    raises ValueError; dtype and range errors follow as_int64.
    """
    values = as_int64(matrix, context="matrix")
    if values.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    a, b, c, d, e, f, g, h, i = (int(value) for value in values.ravel())
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def adjugate_3x3(matrix: object) -> np.ndarray:
    """Compute the transposed cofactor matrix of a 3 by 3 matrix.

    ``matrix`` contains integers accepted by as_int64. Return a readonly int64
    array of shape (3, 3), satisfying matrix @ adjugate == det(matrix) * I.
    Invalid shape raises ValueError; input dtype/range errors follow as_int64.
    Cofactors outside [-INT64_MAX, INT64_MAX] raise ArithmeticRangeError.
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


def unimodular_inverse(matrix: object) -> np.ndarray:
    """Compute the inverse of a unimodular 3 by 3 matrix.

    ``matrix`` contains integers accepted by as_int64 and has determinant
    +1 or -1. Return an int64 array of shape (3, 3), equal to
    adjugate(matrix) / det(matrix). Other determinants or shapes raise
    ValueError; dtype and range errors follow as_int64 and adjugate_3x3.
    """
    determinant = determinant_3x3(matrix)
    if abs(determinant) != 1:
        raise ValueError("matrix must be unimodular")
    return determinant * adjugate_3x3(matrix)


def matmul(left: object, right: object) -> np.ndarray:
    """Compute the matrix product left @ right.

    Operands have shapes (m, k) and (k, n), with integer dtypes accepted by
    as_int64. Return a new int64 array of shape (m, n). Operands are preserved.
    Incompatible shapes raise ValueError and input dtype/range errors follow
    as_int64. Allocation or absolute dot-product bounds exceeding int64 raise
    OverflowError before multiplication. validate_matmul defines the bound.
    """
    a = as_int64(left, context="left operand")
    b = as_int64(right, context="right operand")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible matrix shapes {a.shape} and {b.shape}")
    require_allocation("matrix product", (a.shape[0], b.shape[1]))
    validate_matmul(a, b)
    return _matmul_kernel(a, b)


def validate_matmul(a, b):
    """Validate absolute dot-product bounds before multiplying two matrices.

    ``a`` and ``b`` are integer matrices of shapes (m, k) and (k, n).
    For each output entry, sum(abs(a[i,k] * b[k,j])) must not exceed INT64_MAX.
    This also bounds every product and partial sum. Return None or raise
    OverflowError; callers validate operand types, shapes and allocation.
    """
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            require_bound(
                "matrix product",
                sum(abs(int(a[i, k]) * int(b[k, j])) for k in range(a.shape[1])),
            )


__all__ = [
    "adjugate_3x3",
    "as_int64",
    "determinant_3x3",
    "matmul",
    "to_python_rows",
    "unimodular_inverse",
]


@njit(cache=True)
def _matmul_kernel(a, b):
    """Multiply matrices prepared and validated by matmul.

    ``a`` and ``b`` are int64 arrays of shapes (m, k) and (k, n).
    Return a new int64 array of shape (m, n). matmul must validate input
    ranges, storage size and absolute dot-product bounds before entry.
    """
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                result[i, j] += a[i, k] * b[k, j]
    return result


@njit(cache=True)
def modular_product(a, b, modulus):
    """Multiply compatible integer matrices modulo 1 <= modulus < 2**31.

    Each operand is reduced before multiplication and each partial sum reduced
    immediately. Return a new int64 matrix with canonical nonnegative residues.
    """
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                result[i, j] = (result[i, j] + (a[i, k] % modulus) * (b[k, j] % modulus)) % modulus
    return result


@njit(cache=True)
def gcd(a, b):
    """Return the Euclidean gcd of nonnegative machine-word integers."""
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def bezout(a, b):
    """Return (g, s, t) satisfying s*a + t*b = g = gcd(a, b).

    Congruence callers supply 0 < a < 2**31 and 0 <= b < 2**31. Euclidean
    convergent coefficients alternate signs and are bounded by the inputs;
    these small operands admit all coefficient updates in int64.
    """
    r0, r1, s0, s1, t0, t1 = a, b, 1, 0, 0, 1
    while r1:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
        t0, t1 = t1, t0 - q * t1
    return r0, s0, t0
