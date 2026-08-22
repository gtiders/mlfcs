"""Cartesian tensor actions and invariant-basis numerical kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, qr


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
    dimension = basis.shape[1]
    if dimension == 0:
        return np.empty(0, dtype=np.int32)
    # Preserve the established lexicographic preference for high flattened
    # Cartesian components because it fixes the finite-difference displacement
    # plan.  LAPACK QR supplies each rank test; no numerical elimination is
    # maintained locally.
    selected: list[int] = []
    threshold = tolerance * max(float(np.max(np.abs(basis))), 1.0)
    for row in range(basis.shape[0] - 1, -1, -1):
        trial = basis[np.asarray((*selected, row), dtype=np.int64)]
        r = qr(trial.T, mode="r", pivoting=False, check_finite=False)[0]
        diagonal = np.abs(np.diag(r))
        rank = int(np.count_nonzero(diagonal > threshold))
        if rank > len(selected):
            selected.append(row)
        if len(selected) == dimension:
            break
    if len(selected) != dimension:
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
    symmetric = 0.5 * (np.asarray(gram, dtype=float) + np.asarray(gram, dtype=float).T)
    values, vectors = eigh(symmetric, check_finite=False, driver="evr")
    threshold = tolerance * max(float(np.max(np.abs(values))), 1.0)
    basis = vectors[:, values <= threshold]
    basis[np.abs(basis) < tolerance] = 0.0
    # The second return value historically named the free RREF columns.  It is
    # private and callers only require a stable description of nullity.
    independent = np.arange(basis.shape[1], dtype=np.int32)
    return basis, independent
