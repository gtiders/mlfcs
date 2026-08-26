"""Domain-independent Cartesian tensor actions."""

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


def compose_actions(after: TensorAction, before: TensorAction) -> TensorAction:
    if after.order != before.order:
        raise ValueError("cannot compose tensor actions of different orders")
    return TensorAction(after.rotation @ before.rotation,
                        tuple(before.permutation[i] for i in after.permutation), after.order)


def inverse_action(action: TensorAction) -> TensorAction:
    return TensorAction(np.linalg.inv(action.rotation),
                        tuple(int(value) for value in np.argsort(action.permutation)), action.order)


def apply_action_columns(action: TensorAction, values: np.ndarray) -> np.ndarray:
    return _apply_action_columns_numpy(action, values)
