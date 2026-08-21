from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TensorAction:
    """Host-side Cartesian rotation followed by an IFC-axis permutation.

    Orbit enumeration, constraint construction, and finite-difference
    reconstruction invoke this operation frequently on small, irregular
    tensors.  Keeping it in NumPy avoids device dispatch and transfer costs;
    the fitting-only JAX feature kernels implement their own batched form.
    """

    rotation: np.ndarray
    permutation: tuple[int, ...]
    order: int

    def apply(self, tensor: np.ndarray) -> np.ndarray:
        return _apply_action_tensor_numpy(self, np.asarray(tensor, dtype=float))

    def apply_flat(self, values: np.ndarray) -> np.ndarray:
        tensor = np.asarray(values).reshape((3,) * self.order)
        return self.apply(tensor).reshape(-1)

    def apply_columns(self, values: np.ndarray) -> np.ndarray:
        return _apply_action_columns_numpy(self, values)

    def as_matrix(self) -> np.ndarray:
        return tensor_action_matrix(self.rotation, self.permutation, self.order)


@dataclass(frozen=True, slots=True)
class OrbitImage:
    cluster: tuple[int, ...]
    action: TensorAction


@dataclass(frozen=True, slots=True)
class ClusterOrbit:
    representative: tuple[int, ...]
    basis: np.ndarray
    pivots: np.ndarray
    images: tuple[OrbitImage, ...]

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True, slots=True)
class OrbitSpace:
    order: int
    orbits: tuple[ClusterOrbit, ...]
    cutoff: float
    max_body_order: int | None = None

    @property
    def displacement_keys(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        keys: set[tuple[tuple[int, int], ...]] = set()
        for orbit in self.orbits:
            for flat_component in orbit.pivots:
                components = np.unravel_index(int(flat_component), (3,) * self.order)
                key = tuple(
                    (orbit.representative[axis], int(components[axis]))
                    for axis in range(self.order - 1)
                )
                keys.add(key)
        return tuple(sorted(keys))


def _label_symmetric_basis(cluster: tuple[int, ...]) -> np.ndarray:
    """Return an orthonormal basis invariant under equal-site axis swaps."""
    groups: list[np.ndarray] = []
    for atom in dict.fromkeys(cluster):
        positions = np.asarray(
            [axis for axis, value in enumerate(cluster) if value == atom], dtype=np.int32
        )
        if len(positions) > 1:
            groups.append(positions)

    size = 3 ** len(cluster)
    if not groups:
        return np.eye(size)
    classes: dict[tuple[int, ...], list[int]] = {}
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


def _apply_action_columns_numpy(action: TensorAction, values: np.ndarray) -> np.ndarray:
    """Apply one tensor action without constructing a ``3**order`` square matrix.

    Orbit construction is host-side and calls this only for stabilizers.  A
    NumPy contraction over the already compressed label basis avoids the large
    XLA intermediates produced by vmapping hundreds of sixth-order tensors.
    """
    tensors = np.asarray(values, dtype=float).T.reshape((-1,) + (3,) * action.order)
    transformed = tensors
    for axis in range(action.order):
        transformed = np.tensordot(action.rotation, transformed, axes=((1,), (axis + 1,)))
        transformed = np.moveaxis(transformed, 0, axis + 1)
    axes = (0,) + tuple(axis + 1 for axis in action.permutation)
    return np.transpose(transformed, axes).reshape(len(values.T), -1).T


def _apply_action_tensor_numpy(action: TensorAction, tensor: np.ndarray) -> np.ndarray:
    """Apply one action to one tensor without a dense representation matrix."""
    if tensor.shape != (3,) * action.order:
        raise ValueError(f"expected tensor shape {(3,) * action.order}, got {tensor.shape}")
    transformed = tensor
    for axis in range(action.order):
        transformed = np.tensordot(action.rotation, transformed, axes=((1,), (axis,)))
        transformed = np.moveaxis(transformed, 0, axis)
    return np.transpose(transformed, action.permutation)


def _independent_basis_rows(basis: np.ndarray, tolerance: float) -> np.ndarray:
    """Select deterministic independent tensor components for reconstruction."""
    matrix = basis.T.copy()
    n_rows, n_columns = matrix.shape
    selected: list[int] = []
    pivot_row = 0
    threshold = tolerance * max(float(np.max(np.abs(matrix))), 1.0)
    # Prefer high flattened Cartesian components, matching the free-column
    # convention of the historical full-space RREF while operating on the
    # compressed label-symmetric basis.
    for column in range(n_columns - 1, -1, -1):
        if pivot_row == n_rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(matrix[pivot_row:, column])))
        if abs(matrix[candidate, column]) <= threshold:
            continue
        matrix[[pivot_row, candidate]] = matrix[[candidate, pivot_row]]
        matrix[pivot_row] /= matrix[pivot_row, column]
        for row in range(pivot_row + 1, n_rows):
            if abs(matrix[row, column]) > threshold:
                matrix[row] -= matrix[row, column] * matrix[pivot_row]
        selected.append(column)
        pivot_row += 1
    if len(selected) != basis.shape[1]:
        raise RuntimeError("failed to select independent invariant tensor components")
    return np.asarray(sorted(selected), dtype=np.int32)


def tensor_action_matrix(
    rotation: np.ndarray,
    axis_permutation: tuple[int, ...],
    order: int,
) -> np.ndarray:
    """Return the Cartesian tensor representation of a symmetry action."""
    size = 3**order
    action = TensorAction(np.asarray(rotation, dtype=float), axis_permutation, order)
    return _apply_action_columns_numpy(action, np.eye(size))


def permute_tensor_action(
    action: np.ndarray,
    axis_permutation: tuple[int, ...],
    order: int,
) -> np.ndarray:
    """Permute output axes of a flattened tensor action."""
    size = 3**order
    by_input = action.T.reshape((size,) + (3,) * order)
    axes = (0,) + tuple(axis + 1 for axis in axis_permutation)
    return np.transpose(by_input, axes).reshape(size, size).T


def _null_space_from_gram(gram: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Find null(C) from the small Gram matrix C.T @ C.

    The direct constraint matrix may have tens of thousands of rows for a
    fourth-order high-symmetry cluster. Its Gram matrix is always at most
    81x81, preventing LAPACK from allocating a huge left-singular-vector
    matrix.
    """
    matrix = gram.copy()
    n_rows, n_columns = matrix.shape
    pivot_columns: list[int] = []
    pivot_row = 0
    threshold = tolerance * max(float(np.max(np.abs(matrix))), 1.0)
    for column in range(n_columns):
        if pivot_row == n_rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(matrix[pivot_row:, column])))
        if abs(matrix[candidate, column]) <= threshold:
            continue
        if candidate != pivot_row:
            matrix[[pivot_row, candidate]] = matrix[[candidate, pivot_row]]
        matrix[pivot_row] /= matrix[pivot_row, column]
        for row in range(n_rows):
            if row != pivot_row and abs(matrix[row, column]) > threshold:
                matrix[row] -= matrix[row, column] * matrix[pivot_row]
        matrix[np.abs(matrix) < threshold] = 0.0
        pivot_columns.append(column)
        pivot_row += 1

    independent = np.asarray(
        [column for column in range(n_columns) if column not in pivot_columns],
        dtype=np.int32,
    )
    basis = np.zeros((n_columns, len(independent)), dtype=float)
    for basis_column, free_column in enumerate(independent):
        basis[free_column, basis_column] = 1.0
        for row, dependent_column in enumerate(pivot_columns):
            basis[dependent_column, basis_column] = -matrix[row, free_column]
    basis[np.abs(basis) < tolerance] = 0.0
    return basis, independent


__all__ = [
    "ClusterOrbit",
    "OrbitImage",
    "OrbitSpace",
    "TensorAction",
    "_label_symmetric_basis",
    "permute_tensor_action",
    "tensor_action_matrix",
]
