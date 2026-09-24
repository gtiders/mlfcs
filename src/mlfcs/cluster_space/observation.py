"""Subspace-defined Cartesian component parameters without heuristic thresholds."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.linalg import qr


@njit(cache=True)
def _greedy_rows(orthonormal: np.ndarray) -> np.ndarray:
    """Compiled modified Gram-Schmidt row selection."""
    rows, dimension = orthonormal.shape
    residuals = orthonormal.copy()
    available = np.ones(rows, dtype=np.uint8)
    selected = np.empty(dimension, dtype=np.int32)
    for column in range(dimension):
        best_row = -1
        best_score = -1.0
        for row in range(rows):
            if available[row] == 0:
                continue
            score = 0.0
            for component in range(dimension):
                score += residuals[row, component] * residuals[row, component]
            if score > best_score:
                best_score = score
                best_row = row
        if best_row < 0 or not best_score > 0.0:
            raise ValueError("Cartesian invariant subspace lost its certified dimension")
        selected[column] = best_row
        available[best_row] = 0
        inverse_norm = 1.0 / np.sqrt(best_score)
        direction = np.empty(dimension, dtype=np.float64)
        for component in range(dimension):
            direction[component] = residuals[best_row, component] * inverse_norm
        for row in range(rows):
            overlap = 0.0
            for component in range(dimension):
                overlap += residuals[row, component] * direction[component]
            for component in range(dimension):
                residuals[row, component] -= overlap * direction[component]
    return np.sort(selected)


def component_parameterization(cartesian: object) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(basis, rows, condition)`` with ``basis[rows] == identity``.

    Row selection is greedy max-volume, implemented by incremental orthogonal
    residuals.  No SVD is repeated per candidate and no numerical tolerance
    changes the selected dimension.  ``argmax`` supplies the deterministic
    smallest-index rule for exact ties; the final condition number is reported
    rather than used as another model-selection threshold.
    """
    values = np.asarray(cartesian, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Cartesian basis must be a finite two-dimensional array")
    rows, dimension = values.shape
    if dimension == 0:
        return np.empty((rows, 0)), np.empty(0, dtype=np.int32), 1.0
    orthonormal, _ = qr(values, mode="economic", check_finite=False)
    selected = _greedy_rows(orthonormal)
    observed = orthonormal[selected]
    basis = np.linalg.solve(observed.T, orthonormal.T).T
    condition = float(np.linalg.cond(observed, 2))
    basis.setflags(write=False)
    selected.setflags(write=False)
    return basis, selected, condition


__all__ = ["component_parameterization"]
