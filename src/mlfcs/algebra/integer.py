"""Small exact-integer linear-algebra kernels."""

from __future__ import annotations

import numpy as np
from numba import njit

from mlfcs._arrays import integer_array, require_allocation, require_bound
from mlfcs.errors import IntegerRangeError


def as_int64(values: object, *, context: str) -> np.ndarray:
    """Return exact integers after refusing floating-point rounding and overflow."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{context} must contain integers")
    try:
        return integer_array(array, name=context)
    except OverflowError as error:
        raise IntegerRangeError(f"{context} does not fit the symmetric int64 domain") from error


def determinant_3x3(matrix: object) -> int:
    """Return the exact determinant of an integer 3 by 3 matrix."""
    values = as_int64(matrix, context="matrix")
    if values.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    a, b, c, d, e, f, g, h, i = (int(value) for value in values.ravel())
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def adjugate_3x3(matrix: object) -> np.ndarray:
    """Return the exact adjugate of an integer 3 by 3 matrix."""
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
        return integer_array(np.asarray(result, dtype=object), name="integer adjugate")
    except OverflowError as error:
        raise IntegerRangeError("integer adjugate does not fit in int64") from error


def integer_inverse(matrix: object) -> np.ndarray:
    """Return the exact integer inverse of a unimodular matrix."""
    determinant = determinant_3x3(matrix)
    if abs(determinant) != 1:
        raise ValueError("matrix must be unimodular")
    return determinant * adjugate_3x3(matrix)


def exact_product(left: object, right: object) -> np.ndarray:
    """Multiply integer matrices without permitting silent int64 overflow."""
    a = as_int64(left, context="left operand")
    b = as_int64(right, context="right operand")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible matrix shapes {a.shape} and {b.shape}")
    require_allocation("integer product", (a.shape[0], b.shape[1]))
    prove_integer_product(a, b)
    return integer_product(a, b)


def prove_integer_product(a, b):
    """Certify actual absolute dot products, including each partial sum."""
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            require_bound(
                "integer matrix product",
                sum(abs(int(a[i, k]) * int(b[k, j])) for k in range(a.shape[1])),
            )


def rotate_q_labels(
    labels: object, rotation: object, *, denominator: int | None = None
) -> np.ndarray:
    """Apply a primitive rotation to reciprocal integer labels exactly.

    Fractional positions are row vectors acted on by ``x @ R.T``. Thus a
    reciprocal row label transforms as ``label @ R^{-1}``, not ``R^{-T}``.
    Without ``denominator`` the unreduced image is returned; with it, labels
    are reduced modulo the positive integer denominator of ``q = label / D``.
    """
    values = as_int64(labels, context="reciprocal labels")
    if values.shape[-1:] != (3,):
        raise ValueError("reciprocal labels must end in shape (3,)")
    if denominator is None:
        return exact_product(values.reshape(-1, 3), integer_inverse(rotation)).reshape(values.shape)
    if isinstance(denominator, (bool, np.bool_)) or not isinstance(denominator, (int, np.integer)):
        raise TypeError("denominator must be a positive integer")
    if denominator <= 0:
        raise ValueError("denominator must be a positive integer")
    require_bound("reciprocal modulus", int(denominator), (1 << 31) - 1)
    return modular_product(
        values.reshape(-1, 3), integer_inverse(rotation), int(denominator)
    ).reshape(values.shape)


__all__ = [
    "adjugate_3x3",
    "as_int64",
    "determinant_3x3",
    "exact_product",
    "integer_inverse",
    "rotate_q_labels",
]


@njit(cache=True)
def integer_product(a, b):
    """Multiply compatible int64 matrices after allocation and absolute-dot-product proof.

    Only exact_product should supply admitted operands. Return a new int64
    matrix; unchecked multiply/add loops rely on prove_integer_product.
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
