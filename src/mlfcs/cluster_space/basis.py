"""Exact invariant bases and Cartesian component parameterization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit
from scipy.linalg import qr

from mlfcs._arrays import integer_array, require_allocation
from mlfcs.algebra.exact import exact_kernel
from mlfcs.core.tensors import _tensor_action, prove_tensor_action, tensor_dimension


def _label_symmetric_basis(labels: Sequence[object]) -> np.ndarray:
    """Return the 0/1 tensor basis symmetric between repeated lattice sites."""
    dimension = tensor_dimension(len(labels))
    require_allocation("label basis", (dimension, dimension))
    equal = np.asarray([[a == b for b in labels] for a in labels], dtype=np.uint8).reshape(
        len(labels), len(labels)
    )
    return label_basis(equal)


def _invariant_basis(
    labels: Sequence[object],
    stabilizers: Sequence[tuple[np.ndarray, tuple[int, ...]]],
) -> np.ndarray:
    """Return a saturated int64 basis of the stabilizer invariant space."""
    label_basis = _label_symmetric_basis(labels)
    if stabilizers:
        rotations = integer_array([rotation for rotation, _ in stabilizers])
        permutations = integer_array([permutation for _, permutation in stabilizers])
        if rotations.shape != (len(stabilizers), 3, 3) or permutations.shape != (
            len(stabilizers),
            len(labels),
        ):
            raise ValueError("stabilizer buffers have inconsistent shapes")
        if not np.all(np.sort(permutations, axis=1) == np.arange(len(labels))):
            raise ValueError("stabilizer axes must be permutations")
        for rotation in rotations:
            prove_tensor_action(rotation, len(labels), subtraction=1)
        require_allocation(
            "invariant constraints", (len(stabilizers) * label_basis.shape[0], label_basis.shape[1])
        )
        constraints = invariant_constraints(label_basis, rotations, permutations)
    else:
        constraints = np.empty((0, label_basis.shape[1]), dtype=np.int64)
    constraints = np.unique(constraints, axis=0)
    kernel = exact_kernel(constraints)
    # Each label-basis row selects exactly one column; this avoids an
    # unnecessary integer dot product and its coefficient-growth bound.
    result = kernel[np.argmax(label_basis, axis=1)]
    return result


__all__: list[str] = []


@njit(cache=True)
def label_basis(equal):
    order = equal.shape[0]
    dimension = 3**order
    classes = np.full(dimension, -1, dtype=np.int64)
    column_of_row = np.empty(dimension, dtype=np.int64)
    count = 0
    for row in range(dimension):
        directions = np.empty(order, dtype=np.int64)
        for i in range(order):
            directions[i] = row // (3 ** (order - 1 - i)) % 3
        for i in range(order):
            for j in range(i + 1, order):
                if equal[i, j] and directions[j] < directions[i]:
                    directions[i], directions[j] = directions[j], directions[i]
        key = 0
        for i in range(order):
            key = 3 * key + directions[i]
        if classes[key] < 0:
            classes[key] = count
            count += 1
        column_of_row[row] = classes[key]
    result = np.zeros((dimension, count), dtype=np.int64)
    for row in range(dimension):
        result[row, column_of_row[row]] = 1
    return result


@njit(cache=True)
def invariant_constraints(seed, rotations, permutations):
    rows, columns = seed.shape
    result = np.empty((len(rotations) * rows, columns), dtype=np.int64)
    for operation in range(len(rotations)):
        transformed = _tensor_action(seed, rotations[operation], permutations[operation])
        for i in range(rows):
            for j in range(columns):
                result[operation * rows + i, j] = transformed[i, j] - seed[i, j]
    return result


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
        return np.empty((rows, 0)), np.empty(0, dtype=np.int64), 1.0
    orthonormal, _ = qr(values, mode="economic", check_finite=False)
    selected = select_observation_rows(orthonormal)
    observed = orthonormal[selected]
    basis = np.linalg.solve(observed.T, orthonormal.T).T
    condition = float(np.linalg.cond(observed, 2))
    basis.setflags(write=False)
    selected.setflags(write=False)
    return basis, selected, condition


@njit(cache=True)
def select_observation_rows(orthonormal: np.ndarray) -> np.ndarray:
    """Compiled modified Gram-Schmidt row selection."""
    rows, dimension = orthonormal.shape
    residuals = orthonormal.copy()
    available = np.ones(rows, dtype=np.uint8)
    selected = np.empty(dimension, dtype=np.int64)
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
