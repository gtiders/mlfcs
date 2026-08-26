"""SciPy-backed invariant tensor numerical kernels."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, qr


def invariant_basis_from_gram(gram: np.ndarray, *, tolerance: float) -> np.ndarray:
    """Find the invariant null space from a small constraint Gram matrix."""
    symmetric = 0.5 * (np.asarray(gram, dtype=float) + np.asarray(gram, dtype=float).T)
    values, vectors = eigh(symmetric, check_finite=False, driver="evr")
    threshold = tolerance * max(float(np.max(np.abs(values))), 1.0)
    basis = vectors[:, values <= threshold]
    basis[np.abs(basis) < tolerance] = 0.0
    return basis


def _label_symmetric_basis(cluster: tuple[int, ...]) -> np.ndarray:
    groups = []
    for atom in dict.fromkeys(cluster):
        positions = np.asarray([i for i, value in enumerate(cluster) if value == atom], dtype=np.int32)
        if len(positions) > 1:
            groups.append(positions)
    size = 3 ** len(cluster)
    if not groups:
        return np.eye(size)
    classes = {}
    for flat, component in enumerate(np.ndindex((3,) * len(cluster))):
        canonical = list(component)
        for positions in groups:
            ordered = sorted(canonical[int(position)] for position in positions)
            for position, direction in zip(positions, ordered, strict=True):
                canonical[int(position)] = direction
        classes.setdefault(tuple(canonical), []).append(flat)
    basis = np.zeros((size, len(classes)))
    for column, members in enumerate(classes.values()):
        basis[members, column] = 1.0 / np.sqrt(len(members))
    return basis

def select_independent_rows(
    basis: np.ndarray, *, tolerance: float
) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.empty(0, dtype=np.int32)
    selected = []
    threshold = tolerance * max(float(np.max(np.abs(basis))), 1.0)
    for row in range(basis.shape[0] - 1, -1, -1):
        trial = basis[np.asarray((*selected, row), dtype=np.int64)]
        diagonal = np.abs(np.diag(qr(trial.T, mode="r", pivoting=False, check_finite=False)[0]))
        if int(np.count_nonzero(diagonal > threshold)) > len(selected):
            selected.append(row)
        if len(selected) == basis.shape[1]:
            break
    if len(selected) != basis.shape[1]:
        raise RuntimeError("failed to select independent invariant tensor components")
    return np.asarray(sorted(selected), dtype=np.int32)


def normalize_pivot_basis(basis: np.ndarray, pivots: np.ndarray) -> np.ndarray:
    return np.linalg.solve(basis[pivots].T, basis.T).T


__all__ = ["invariant_basis_from_gram", "normalize_pivot_basis", "select_independent_rows"]
