"""Small exact-integer linear-algebra kernels."""

from __future__ import annotations

import numpy as np

from mlfcs.core.errors import IntegerRangeError

_INT64_SAFE_BOUND = 2**62


def as_int64(values: object, *, context: str) -> np.ndarray:
    """Return exact integers after refusing floating-point rounding and overflow."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{context} must contain integers")
    objects = array.astype(object)
    limit = np.iinfo(np.int64)
    if any(int(value) < limit.min or int(value) > limit.max for value in objects.flat):
        raise IntegerRangeError(f"{context} does not fit in int64")
    return np.asarray(objects, dtype=np.int64)


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
        return np.asarray(result, dtype=np.int64)
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
    bound = (
        int(a.shape[1])
        * (int(np.max(np.abs(a.astype(object)), initial=0)) if a.size else 0)
        * (int(np.max(np.abs(b.astype(object)), initial=0)) if b.size else 0)
    )
    if bound < _INT64_SAFE_BOUND:
        return a @ b
    return np.asarray(a.astype(object) @ b.astype(object))


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
    moved = exact_product(values.reshape(-1, 3), integer_inverse(rotation)).reshape(values.shape)
    if denominator is None:
        return moved
    if isinstance(denominator, (bool, np.bool_)) or not isinstance(denominator, (int, np.integer)):
        raise TypeError("denominator must be a positive integer")
    if denominator <= 0:
        raise ValueError("denominator must be a positive integer")
    return np.mod(moved, int(denominator))


__all__ = [
    "adjugate_3x3",
    "as_int64",
    "determinant_3x3",
    "exact_product",
    "integer_inverse",
    "rotate_q_labels",
]
