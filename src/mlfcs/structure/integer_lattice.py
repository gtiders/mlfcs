"""Small exact-integer helpers for three-dimensional lattice quotients."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form


def normalize_supercell_matrix(matrix: object) -> np.ndarray:
    """Return a validated full-rank integer 3x3 supercell matrix."""
    values = np.asarray(matrix)
    if values.shape == (3,):
        values = np.diag(values)
    if values.shape != (3, 3):
        raise ValueError("supercell_matrix must be three repeats or an integer 3x3 matrix")
    if np.issubdtype(values.dtype, np.integer):
        integers = [[int(value) for value in row] for row in values]
    else:
        rounded = np.rint(values)
        if not np.allclose(values, rounded, atol=1e-10, rtol=0.0):
            raise ValueError("supercell_matrix must contain integers")
        integers = [[int(value) for value in row] for row in rounded]
    limit = np.iinfo(np.int64)
    if any(value < limit.min or value > limit.max for row in integers for value in row):
        raise OverflowError("supercell_matrix does not fit in int64")
    rounded = np.asarray(integers, dtype=np.int64)
    if determinant_3x3(rounded) == 0:
        raise ValueError("supercell_matrix must be nonsingular")
    return rounded


def determinant_3x3(matrix: np.ndarray) -> int:
    """Return the exact determinant of an integer 3x3 matrix."""
    values = np.asarray(matrix, dtype=np.int64)
    if values.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    a, b, c, d, e, f, g, h, i = (int(value) for value in values.ravel())
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def adjugate_3x3(matrix: np.ndarray) -> np.ndarray:
    """Return the exact integer adjugate of a 3x3 matrix."""
    values = np.asarray(matrix, dtype=np.int64)
    if values.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    a, b, c, d, e, f, g, h, i = (int(value) for value in values.ravel())
    result = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    limit = np.iinfo(np.int64)
    if any(value < limit.min or value > limit.max for row in result for value in row):
        raise OverflowError("integer adjugate does not fit in int64")
    return np.asarray(result, dtype=np.int64)


def residue_key(translation: np.ndarray, matrix: np.ndarray) -> tuple[int, int, int]:
    """Return the exact key of a translation in ``Z^3 / Z^3 S``."""
    vector = np.asarray(translation, dtype=np.int64)
    if vector.shape != (3,):
        raise ValueError("translation must have shape (3,)")
    determinant = abs(determinant_3x3(matrix))
    if determinant == 0:
        raise ValueError("matrix must be nonsingular")
    adjugate = adjugate_3x3(matrix)
    residue = [
        sum(int(vector[row]) * int(adjugate[row, column]) for row in range(3)) % determinant
        for column in range(3)
    ]
    return tuple(int(value) for value in residue)


def same_residue(translation_a: np.ndarray, translation_b: np.ndarray, matrix: np.ndarray) -> bool:
    """Return whether two integer translations belong to the same residue."""
    return residue_key(
        np.asarray(translation_a, dtype=np.int64) - np.asarray(translation_b, dtype=np.int64),
        matrix,
    ) == (0, 0, 0)


def _sympy_int64_matrix(matrix: Matrix) -> np.ndarray:
    """Convert a SymPy integer matrix after an explicit int64 bounds check."""
    limit = np.iinfo(np.int64)
    values = [[int(value) for value in row] for row in matrix.tolist()]
    if any(value < limit.min or value > limit.max for row in values for value in row):
        raise OverflowError("integer normal form does not fit in int64")
    return np.asarray(values, dtype=np.int64)


def row_hermite_normal_form(matrix: np.ndarray) -> np.ndarray:
    """Return the canonical lower-triangular row HNF.

    SymPy defines a canonical column Hermite normal form.  Applying that
    implementation to ``S.T`` and transposing the result gives the row form
    for the row lattice used by MLFCS translations.
    """
    values = normalize_supercell_matrix(matrix).astype(np.int64)
    source = Matrix([[int(value) for value in row] for row in values])
    hnf_sympy = hermite_normal_form(source.T).T
    hnf = _sympy_int64_matrix(hnf_sympy)
    if np.any(np.diag(hnf) <= 0) or np.any(np.triu(hnf, 1) != 0):
        raise RuntimeError("row HNF does not use the expected lower-triangular convention")
    return hnf


@dataclass(frozen=True, slots=True)
class IntegerLatticeQuotient:
    """Canonical quotient ``Z^3 / Z^3 S`` backed by row HNF.

    Translations are row vectors.  SymPy's canonical row HNF is lower
    triangular, so reduction proceeds from the last coordinate to the first.
    The resulting fundamental-domain remainder satisfies
    ``0 <= r[i] < H[i, i]``.
    """

    matrix: np.ndarray
    hnf: np.ndarray = field(init=False)
    representatives: np.ndarray = field(init=False)
    _strides: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        matrix = normalize_supercell_matrix(self.matrix)
        hnf = row_hermite_normal_form(matrix)
        diagonal = tuple(int(value) for value in np.diag(hnf))
        representatives = np.asarray(
            tuple(product(*(range(value) for value in diagonal))), dtype=np.int64
        )
        representatives = representatives.reshape((-1, 3))
        expected = abs(determinant_3x3(matrix))
        if len(representatives) != expected:
            raise RuntimeError("HNF fundamental domain size differs from the supercell determinant")
        strides = np.asarray((diagonal[1] * diagonal[2], diagonal[2], 1), dtype=np.int64)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "hnf", hnf)
        object.__setattr__(self, "representatives", representatives)
        object.__setattr__(self, "_strides", strides)

    @property
    def size(self) -> int:
        return len(self.representatives)

    def decompose(self, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return exact ``q, r`` such that ``translation = q @ H + r``."""
        values = np.asarray(translation)
        if values.shape != (3,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("translation must be an integer vector with shape (3,)")
        quotient, remainder = self.decompose_many(values.reshape(1, 3))
        return quotient[0], remainder[0]

    def decompose_many(self, translations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized exact HNF decomposition of integer translations."""
        values = np.asarray(translations)
        if values.ndim < 1 or values.shape[-1] != 3 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("translations must be an integer array ending in shape (3,)")
        original_shape = values.shape
        flattened = values.astype(np.int64, copy=False).reshape(-1, 3)
        remainder = flattened.copy()
        quotient = np.zeros_like(remainder)
        for axis in range(2, -1, -1):
            coefficient = np.floor_divide(remainder[:, axis], self.hnf[axis, axis])
            quotient[:, axis] = coefficient
            remainder -= coefficient[:, None] * self.hnf[axis]
        if not np.array_equal(quotient @ self.hnf + remainder, flattened):
            raise RuntimeError("HNF quotient decomposition failed")
        return quotient.reshape(original_shape), remainder.reshape(original_shape)

    def reduce(self, translation: np.ndarray) -> np.ndarray:
        """Return the canonical HNF fundamental-domain representative."""
        return self.decompose(translation)[1]

    def reduce_many(self, translations: np.ndarray) -> np.ndarray:
        """Return canonical representatives for an array of translations."""
        return self.decompose_many(translations)[1]

    def cell_index(self, translation: np.ndarray) -> int:
        """Return the deterministic lexicographic HNF cell index."""
        return int(self.cell_index_many(np.asarray(translation).reshape(1, 3))[0])

    def cell_index_many(self, translations: np.ndarray) -> np.ndarray:
        """Return mixed-radix HNF cell indices for many translations."""
        remainders = self.reduce_many(translations)
        return np.sum(remainders * self._strides, axis=-1, dtype=np.int64)

    def equivalent(self, translation_a: np.ndarray, translation_b: np.ndarray) -> bool:
        """Return whether two translations are in the same quotient class."""
        return np.array_equal(self.reduce(translation_a), self.reduce(translation_b))


__all__ = [
    "IntegerLatticeQuotient",
    "adjugate_3x3",
    "determinant_3x3",
    "normalize_supercell_matrix",
    "residue_key",
    "row_hermite_normal_form",
    "same_residue",
]
