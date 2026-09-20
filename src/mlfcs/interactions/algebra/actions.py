"""Domain-independent tensor actions in the Cartesian and lattice frames."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np


@dataclass(frozen=True, slots=True)
class TensorAction:
    r"""Host-side rotation followed by an IFC-axis permutation.

    ``rotation`` is the Cartesian rotation that acts on the fitted tensor components.
    ``scaled_rotation`` is the same operation on lattice (scaled) coordinates, where a
    spglib rotation is an integer matrix.  Orbit constraints and identifiability are
    decided from the scaled frame, because a cell that is not aligned with the
    reference axes (an fcc primitive $60^\circ$ cell, hexagonal and rhombohedral cells)
    has irrational entries in the Cartesian rotation only.

    Orbit enumeration, constraint construction, and finite-difference reconstruction
    invoke this operation frequently on small, irregular tensors.  Keeping it in NumPy
    avoids device dispatch and transfer costs; the fitting-only JAX feature kernels
    implement their own batched form.
    """

    rotation: np.ndarray
    permutation: tuple[int, ...]
    order: int
    scaled_rotation: np.ndarray | None = None

    def apply(self, tensor: np.ndarray) -> np.ndarray:
        return _apply_action_tensor_numpy(self, np.asarray(tensor, dtype=float))

    def apply_flat(self, values: np.ndarray) -> np.ndarray:
        tensor = np.asarray(values).reshape((3,) * self.order)
        return self.apply(tensor).reshape(-1)

    def apply_columns(self, values: np.ndarray) -> np.ndarray:
        return _apply_action_columns_numpy(self, values)

    def apply_scaled_columns(self, values: np.ndarray) -> np.ndarray:
        return _apply_scaled_columns_numpy(self, values)

    def as_matrix(self) -> np.ndarray:
        return tensor_action_matrix(self.rotation, self.permutation, self.order)

    def as_scaled_matrix(self) -> np.ndarray:
        return scaled_action_matrix(self)


def _apply_action_columns_numpy(action: TensorAction, values: np.ndarray) -> np.ndarray:
    """Apply one tensor action without constructing a ``3**order`` square matrix.

    Orbit construction is host-side and calls this only for stabilizers.  A
    NumPy contraction over the already compressed label basis avoids the large
    XLA intermediates produced by vmapping hundreds of sixth-order tensors.
    """
    values = np.asarray(values, dtype=float)
    if values.shape[1] == 0:
        return np.empty((3**action.order, 0), dtype=float)
    tensors = values.T.reshape((-1,) + (3,) * action.order)
    transformed = tensors
    for axis in range(action.order):
        transformed = np.tensordot(action.rotation, transformed, axes=((1,), (axis + 1,)))
        transformed = np.moveaxis(transformed, 0, axis + 1)
    axes = (0,) + tuple(axis + 1 for axis in action.permutation)
    return np.transpose(transformed, axes).reshape(len(values.T), -1).T


def _apply_scaled_columns_numpy(action: TensorAction, values: np.ndarray) -> np.ndarray:
    """Apply the lattice-frame rotation of ``action`` to integer tensor columns."""
    return apply_scaled_columns(action.scaled_rotation, action.permutation, action.order, values)


def apply_scaled_columns(
    rotation: np.ndarray | None,
    permutation: tuple[int, ...],
    order: int,
    values: np.ndarray,
) -> np.ndarray:
    """Apply one lattice-frame rotation to integer tensor columns, exactly."""
    if rotation is None:
        raise ValueError("action carries no lattice-frame rotation")
    values = np.asarray(values, dtype=np.int64)
    if values.shape[1] == 0:
        return np.empty((3**order, 0), dtype=np.int64)
    rotation = np.asarray(rotation, dtype=np.int64)
    if rotation.shape != (3, 3):
        raise ValueError(f"expected a 3x3 lattice rotation, got {rotation.shape}")
    tensors = values.T.reshape((-1,) + (3,) * order)
    transformed = tensors
    for axis in range(order):
        transformed = np.tensordot(rotation, transformed, axes=((1,), (axis + 1,)))
        transformed = np.moveaxis(transformed, 0, axis + 1)
    axes = (0,) + tuple(axis + 1 for axis in permutation)
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


def tensor_action_matrix(
    rotation: np.ndarray,
    axis_permutation: tuple[int, ...],
    order: int,
) -> np.ndarray:
    """Return the Cartesian tensor representation of a symmetry action."""
    size = 3**order
    action = TensorAction(np.asarray(rotation, dtype=float), axis_permutation, order)
    return _apply_action_columns_numpy(action, np.eye(size))


def scaled_action_matrix(action: TensorAction) -> np.ndarray:
    """Return the exact integer lattice-frame tensor representation of one action."""
    size = 3**action.order
    return apply_scaled_columns(
        action.scaled_rotation, action.permutation, action.order, np.eye(size, dtype=np.int64)
    )


def scaled_to_cartesian_rotation(scaled_rotation: np.ndarray, cell: np.ndarray) -> np.ndarray:
    r"""Return the Cartesian rotation $\text{cell}^T R \,\text{cell}^{-T}$ of a cell."""
    scaled = np.asarray(scaled_rotation, dtype=np.int64)
    if scaled.shape != (3, 3):
        raise ValueError(f"expected a 3x3 lattice rotation, got {scaled.shape}")
    lattice = np.asarray(cell, dtype=float)
    return lattice.T @ scaled @ np.linalg.inv(lattice.T)


def scaled_to_cartesian_matrix(cell: np.ndarray, order: int) -> np.ndarray:
    r"""Return $K = (\text{cell}^T)^{\otimes\, \text{order}}$ for tensor components.

    With this map the two frames agree, ``K @ scaled @ inverse(K)`` being the Cartesian
    tensor representation of the same operation, so a lattice-frame invariant tensor
    has the Cartesian components ``K @ values``.
    """
    factor = np.asarray(cell, dtype=float).T
    matrix = np.ones((1, 1), dtype=float)
    for _ in range(order):
        matrix = np.kron(matrix, factor)
    return matrix


def compose_actions(after: TensorAction, before: TensorAction) -> TensorAction:
    if after.order != before.order:
        raise ValueError("cannot compose tensor actions of different orders")
    scaled = None
    if after.scaled_rotation is not None and before.scaled_rotation is not None:
        scaled = after.scaled_rotation @ before.scaled_rotation
    return TensorAction(
        after.rotation @ before.rotation,
        tuple(before.permutation[i] for i in after.permutation),
        after.order,
        scaled,
    )


def inverse_action(action: TensorAction) -> TensorAction:
    scaled = None
    if action.scaled_rotation is not None:
        scaled = _cached_integer_inverse(
            np.ascontiguousarray(action.scaled_rotation, dtype=np.int64).tobytes()
        )
    return TensorAction(
        np.linalg.inv(action.rotation),
        tuple(int(value) for value in np.argsort(action.permutation)),
        action.order,
        scaled,
    )


@cache
def _cached_integer_inverse(rotation: bytes) -> np.ndarray | None:
    return _integer_inverse(np.frombuffer(rotation, dtype=np.int64).reshape(3, 3).copy())


def _integer_inverse(matrix: np.ndarray) -> np.ndarray | None:
    """Return the integer inverse of an integer matrix, when it is integral."""
    candidate = np.rint(np.linalg.inv(np.asarray(matrix, dtype=float))).astype(np.int64)
    if np.array_equal(candidate @ np.asarray(matrix, dtype=np.int64), np.eye(3, dtype=np.int64)):
        return candidate
    return None


def apply_action_columns(action: TensorAction, values: np.ndarray) -> np.ndarray:
    return _apply_action_columns_numpy(action, values)
