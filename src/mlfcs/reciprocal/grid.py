"""Exact reciprocal characters of finite supercell translation groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.structure.integer_lattice import (
    IntegerLatticeQuotient,
    adjugate_3x3,
    determinant_3x3,
    normalize_supercell_matrix,
)


@dataclass(frozen=True, slots=True)
class ReciprocalQuotientGrid:
    """Integer labels and fractional coordinates for reciprocal characters."""

    labels: np.ndarray
    denominator: int
    points: np.ndarray

    def negative_label(self, label: np.ndarray) -> tuple[int, int, int]:
        values = np.mod(-np.asarray(label, dtype=np.int64), self.denominator)
        return tuple(int(value) for value in values)


def reciprocal_quotient_grid(integer_matrix: object) -> ReciprocalQuotientGrid:
    """Return the reciprocal quotient in deterministic exact-label order."""
    matrix = normalize_supercell_matrix(integer_matrix)
    determinant = abs(determinant_3x3(matrix))
    representatives = IntegerLatticeQuotient(matrix.T).representatives
    numerators = representatives @ adjugate_3x3(matrix).T
    labels = np.mod(numerators, determinant).astype(np.int64)
    points = labels.astype(float) / determinant
    if len(np.unique(labels, axis=0)) != determinant:
        raise RuntimeError("reciprocal quotient contains duplicate q points")
    if not np.allclose(points @ matrix.T, np.rint(points @ matrix.T), atol=1e-12, rtol=0.0):
        raise RuntimeError("reciprocal quotient contains an incompatible q point")
    return ReciprocalQuotientGrid(labels, determinant, points)


def quotient_qpoints(integer_matrix: object) -> np.ndarray:
    """Return floating q coordinates in exact quotient-label order."""
    return reciprocal_quotient_grid(integer_matrix).points


__all__ = ["ReciprocalQuotientGrid", "quotient_qpoints", "reciprocal_quotient_grid"]
