"""Small exact-integer helpers for three-dimensional lattice quotients."""

from __future__ import annotations

import numpy as np


def normalize_supercell_matrix(matrix: object) -> np.ndarray:
    """Return a validated full-rank integer 3x3 supercell matrix."""
    values = np.asarray(matrix)
    if values.shape == (3,):
        values = np.diag(values)
    if values.shape != (3, 3):
        raise ValueError("supercell_matrix must be three repeats or an integer 3x3 matrix")
    rounded = np.rint(values).astype(np.int64)
    if not np.allclose(values, rounded, atol=1e-10, rtol=0.0):
        raise ValueError("supercell_matrix must contain integers")
    if determinant_3x3(rounded) == 0:
        raise ValueError("supercell_matrix must be nonsingular")
    return rounded.astype(np.int32)


def determinant_3x3(matrix: np.ndarray) -> int:
    """Return the exact determinant of an integer 3x3 matrix."""
    values = np.asarray(matrix, dtype=np.int64)
    if values.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    a, b, c = values[0]
    d, e, f = values[1]
    g, h, i = values[2]
    return int(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))


def adjugate_3x3(matrix: np.ndarray) -> np.ndarray:
    """Return the exact integer adjugate of a 3x3 matrix."""
    values = np.asarray(matrix, dtype=np.int64)
    if values.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    a, b, c = values[0]
    d, e, f = values[1]
    g, h, i = values[2]
    return np.asarray(
        (
            (e * i - f * h, c * h - b * i, b * f - c * e),
            (f * g - d * i, a * i - c * g, c * d - a * f),
            (d * h - e * g, b * g - a * h, a * e - b * d),
        ),
        dtype=np.int64,
    )


def residue_key(translation: np.ndarray, matrix: np.ndarray) -> tuple[int, int, int]:
    """Return the exact key of a translation in ``Z^3 / Z^3 S``."""
    vector = np.asarray(translation, dtype=np.int64)
    if vector.shape != (3,):
        raise ValueError("translation must have shape (3,)")
    determinant = abs(determinant_3x3(matrix))
    if determinant == 0:
        raise ValueError("matrix must be nonsingular")
    residue = np.mod(vector @ adjugate_3x3(matrix), determinant)
    return tuple(int(value) for value in residue)


def same_residue(
    translation_a: np.ndarray, translation_b: np.ndarray, matrix: np.ndarray
) -> bool:
    """Return whether two integer translations belong to the same residue."""
    return residue_key(
        np.asarray(translation_a, dtype=np.int64)
        - np.asarray(translation_b, dtype=np.int64),
        matrix,
    ) == (0, 0, 0)


__all__ = [
    "adjugate_3x3",
    "determinant_3x3",
    "normalize_supercell_matrix",
    "residue_key",
    "same_residue",
]
