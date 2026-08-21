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


def row_hermite_normal_form(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical lower-triangular row HNF ``H = U @ S``.

    SymPy defines a canonical column Hermite normal form.  Applying that
    implementation to ``S.T`` and transposing the result gives the row form
    for the row lattice used by MLFCS translations.  The transformation is
    recovered exactly and verified to be integer and unimodular.
    """
    values = normalize_supercell_matrix(matrix).astype(np.int64)
    source = Matrix([[int(value) for value in row] for row in values])
    hnf_sympy = hermite_normal_form(source.T).T
    transform_sympy = hnf_sympy * source.inv()
    if any(not value.is_Integer for value in transform_sympy):
        raise RuntimeError("row HNF transformation is not integral")
    hnf = np.asarray(hnf_sympy.tolist(), dtype=np.int64)
    transform = np.asarray(transform_sympy.tolist(), dtype=np.int64)
    if not np.array_equal(transform @ values, hnf):
        raise RuntimeError("row HNF transformation does not reproduce the normal form")
    if abs(determinant_3x3(transform)) != 1:
        raise RuntimeError("row HNF transformation is not unimodular")
    if np.any(np.diag(hnf) <= 0) or np.any(np.triu(hnf, 1) != 0):
        raise RuntimeError("row HNF does not use the expected lower-triangular convention")
    return hnf, transform


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
    transformation: np.ndarray = field(init=False)
    representatives: np.ndarray = field(init=False)
    _cell_by_representative: dict[tuple[int, int, int], int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        matrix = normalize_supercell_matrix(self.matrix)
        hnf, transformation = row_hermite_normal_form(matrix)
        diagonal = tuple(int(value) for value in np.diag(hnf))
        representatives = np.asarray(tuple(product(*(range(value) for value in diagonal))), dtype=np.int64)
        representatives = representatives.reshape((-1, 3))
        expected = abs(determinant_3x3(matrix))
        if len(representatives) != expected:
            raise RuntimeError("HNF fundamental domain size differs from the supercell determinant")
        lookup = {
            tuple(int(value) for value in representative): cell
            for cell, representative in enumerate(representatives)
        }
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "hnf", hnf.astype(np.int32))
        object.__setattr__(self, "transformation", transformation.astype(np.int32))
        object.__setattr__(self, "representatives", representatives.astype(np.int32))
        object.__setattr__(self, "_cell_by_representative", lookup)

    @property
    def size(self) -> int:
        return len(self.representatives)

    def decompose(self, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return exact ``q, r`` such that ``translation = q @ H + r``."""
        values = np.asarray(translation)
        if values.shape != (3,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("translation must be an integer vector with shape (3,)")
        remainder = [int(value) for value in values]
        quotient = [0, 0, 0]
        hnf = np.asarray(self.hnf, dtype=np.int64)
        for axis in range(2, -1, -1):
            divisor = int(hnf[axis, axis])
            coefficient = remainder[axis] // divisor
            quotient[axis] = coefficient
            for component in range(3):
                remainder[component] -= coefficient * int(hnf[axis, component])
        q = np.asarray(quotient, dtype=np.int64)
        r = np.asarray(remainder, dtype=np.int64)
        if not np.array_equal(q @ hnf + r, values.astype(np.int64)):
            raise RuntimeError("HNF quotient decomposition failed")
        return q, r

    def reduce(self, translation: np.ndarray) -> np.ndarray:
        """Return the canonical HNF fundamental-domain representative."""
        return self.decompose(translation)[1].astype(np.int32)

    def cell_index(self, translation: np.ndarray) -> int:
        """Return the deterministic lexicographic HNF cell index."""
        return self._cell_by_representative[tuple(int(value) for value in self.reduce(translation))]

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
