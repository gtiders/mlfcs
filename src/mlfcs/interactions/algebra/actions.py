"""Domain-independent tensor actions in the Cartesian and lattice frames.

One action carries both frames of the same symmetry operation.  ``rotation`` is the
Cartesian rotation that acts on fitted tensor components.  ``scaled_rotation`` is the
spglib lattice (scaled) rotation, which is an integer matrix for *every* cell: an fcc
primitive 60 degree cell, hexagonal and rhombohedral cells have irrational entries in
their Cartesian rotation only.  Orbit constraints, invariant kernels, identifiability
and coefficient provenance are decided from the lattice frame, where nothing has to be
rounded; the Cartesian rotation is used only to render physical tensors.

Orbit enumeration, constraint construction, and finite-difference reconstruction invoke
this operation frequently on small, irregular tensors.  Keeping it in NumPy avoids
device dispatch and transfer costs; the compiled fitting kernels implement their own
batched Cartesian form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.exceptions import IntegerRangeError
from mlfcs.structure.integer_lattice import adjugate_3x3, determinant_3x3


@dataclass(frozen=True, slots=True)
class TensorAction:
    """Host-side rotation followed by an IFC-axis permutation."""

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
        """Apply the lattice-frame rotation to integer tensor columns, exactly."""
        return apply_lattice_columns(self.scaled_rotation, self.permutation, self.order, values)


def _apply_action_columns_numpy(action: TensorAction, values: np.ndarray) -> np.ndarray:
    """Apply one Cartesian tensor action to a block of component columns.

    A NumPy contraction over the already compressed label basis avoids the large
    array intermediates produced by vmapping hundreds of high-order tensors.
    """
    values = np.asarray(values, dtype=float)
    if values.shape[1] == 0:
        return np.empty((3**action.order, 0), dtype=float)
    tensors = values.T.reshape((-1,) + (3,) * action.order)
    transformed = _rotate_tensor_axes(tensors, action.rotation, action.order, offset=1)
    axes = (0,) + tuple(axis + 1 for axis in action.permutation)
    return np.transpose(transformed, axes).reshape(len(values.T), -1).T


def _apply_action_tensor_numpy(action: TensorAction, tensor: np.ndarray) -> np.ndarray:
    """Apply one action to one tensor without a dense representation matrix."""
    if tensor.shape != (3,) * action.order:
        raise ValueError(f"expected tensor shape {(3,) * action.order}, got {tensor.shape}")
    transformed = _rotate_tensor_axes(tensor, action.rotation, action.order, offset=0)
    return np.transpose(transformed, action.permutation)


def _rotate_tensor_axes(
    tensors: np.ndarray, rotation: np.ndarray, order: int, *, offset: int
) -> np.ndarray:
    """Contract every tensor axis with ``rotation``, keeping the axis order.

    ``offset`` is 1 for a leading batch axis of tensor components and 0 for one tensor,
    so the contraction always addresses the axis being rotated.
    """
    transformed = tensors
    for axis in range(order):
        transformed = np.tensordot(rotation, transformed, axes=((1,), (axis + offset,)))
        transformed = np.moveaxis(transformed, 0, axis + offset)
    return transformed


def apply_lattice_columns(
    rotation: np.ndarray | None,
    permutation: tuple[int, ...],
    order: int,
    values: np.ndarray,
) -> np.ndarray:
    """Apply one lattice-frame rotation to integer tensor columns, exactly.

    The lattice rotation of a symmetry operation is spglib's scaled rotation, an
    integer matrix for every cell, so this path is exact integer arithmetic and
    needs no tolerance.
    """
    if rotation is None:
        raise ValueError("action carries no lattice-frame rotation")
    lattice = as_int64(np.asarray(rotation), context="lattice rotation")
    if lattice.shape != (3, 3):
        raise ValueError(f"expected a 3x3 lattice rotation, got {lattice.shape}")
    columns = as_int64(np.asarray(values), context="lattice tensor columns")
    if columns.shape[0] != 3**order:
        raise ValueError(f"expected {3**order} tensor components, got {columns.shape[0]}")
    if columns.shape[1] == 0:
        return np.empty((3**order, 0), dtype=np.int64)
    tensors = columns.T.reshape((-1,) + (3,) * order)
    transformed = _rotate_tensor_axes(tensors, lattice, order, offset=1)
    axes = (0,) + tuple(axis + 1 for axis in permutation)
    return np.ascontiguousarray(np.transpose(transformed, axes).reshape(len(columns.T), -1).T)


def scaled_to_cartesian_rotation(scaled_rotation: np.ndarray, cell: np.ndarray) -> np.ndarray:
    r"""Return the Cartesian rotation of one lattice rotation for a cell.

    With the row-vector convention used repo-wide (``fractional @ cell`` is Cartesian)
    this is the transpose of ``inv(cell) @ R.T @ cell``, which is the Cartesian
    rotation acting on tensor components the way ``TensorAction.apply_columns`` expects.
    """
    scaled = as_int64(np.asarray(scaled_rotation), context="lattice rotation")
    if scaled.shape != (3, 3):
        raise ValueError(f"expected a 3x3 lattice rotation, got {scaled.shape}")
    lattice = np.asarray(cell, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"expected a 3x3 cell, got {lattice.shape}")
    return lattice.T @ scaled @ np.linalg.inv(lattice.T)


def integer_inverse(rotation: np.ndarray) -> np.ndarray:
    """Return the exact integer inverse of a unimodular 3x3 integer matrix.

    Symmetry rotations are integral and orthogonal over the lattice, so their inverse is
    integral as well; computing it exactly keeps the lattice frame closed under
    composition instead of falling back to a floating point inverse.
    """
    lattice = as_int64(np.asarray(rotation), context="lattice rotation")
    if lattice.shape != (3, 3):
        raise ValueError(f"expected a 3x3 lattice rotation, got {lattice.shape}")
    determinant = determinant_3x3(lattice)
    if abs(determinant) != 1:
        raise ValueError(f"lattice rotation is not unimodular (determinant {determinant})")
    adjugate = _adjugate_or_raise(lattice)
    return np.ascontiguousarray(np.asarray(adjugate, dtype=np.int64) * int(np.sign(determinant)))


def _adjugate_or_raise(matrix: np.ndarray) -> np.ndarray:
    """Return the integer adjugate, reporting an explicit range failure."""
    try:
        return adjugate_3x3(matrix)
    except OverflowError as error:
        raise IntegerRangeError(
            "the integer adjugate of a lattice rotation does not fit in int64; "
            "reduce the primitive cell before building orbit algebra"
        ) from error


def as_int64(values: np.ndarray, *, context: str) -> np.ndarray:
    """Return integer values as a contiguous int64 array, or raise a readable error.

    Arbitrary-size Python integers are accepted only while they fit int64; a wider
    value reports the affected quantity instead of letting NumPy truncate it or a C
    extension raise an unrelated conversion error.
    """
    array = np.asarray(values)
    limit = np.iinfo(np.int64)
    if array.dtype == object:
        flat = [int(value) for value in array.reshape(-1)]
        if any(value < limit.min or value > limit.max for value in flat):
            raise IntegerRangeError(
                f"{context} does not fit in int64; reduce the primitive cell before "
                "building orbit algebra in the lattice frame"
            )
        return np.ascontiguousarray(np.asarray(flat, dtype=np.int64).reshape(array.shape))
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{context} must contain integers, got dtype {array.dtype}")
    if array.dtype == np.uint64 and array.size and int(array.max()) > limit.max:
        raise IntegerRangeError(f"{context} does not fit in int64")
    return np.ascontiguousarray(array.astype(np.int64, copy=False))


def compose_actions(after: TensorAction, before: TensorAction) -> TensorAction:
    if after.order != before.order:
        raise ValueError("cannot compose tensor actions of different orders")
    scaled = None
    if (after.scaled_rotation is None) != (before.scaled_rotation is None):
        raise ValueError("cannot compose tensor actions from different frames")
    if after.scaled_rotation is not None:
        scaled = as_int64(
            np.asarray(after.scaled_rotation, dtype=np.int64)
            @ np.asarray(before.scaled_rotation, dtype=np.int64),
            context="composed lattice rotation",
        )
    return TensorAction(
        after.rotation @ before.rotation,
        tuple(before.permutation[i] for i in after.permutation),
        after.order,
        scaled,
    )


def inverse_action(action: TensorAction) -> TensorAction:
    scaled = None
    if action.scaled_rotation is not None:
        scaled = integer_inverse(action.scaled_rotation)
    return TensorAction(
        np.linalg.inv(action.rotation),
        tuple(int(value) for value in np.argsort(action.permutation)),
        action.order,
        scaled,
    )


def apply_action_columns(action: TensorAction, values: np.ndarray) -> np.ndarray:
    return _apply_action_columns_numpy(action, values)


__all__ = [
    "TensorAction",
    "apply_action_columns",
    "apply_lattice_columns",
    "as_int64",
    "compose_actions",
    "integer_inverse",
    "inverse_action",
    "scaled_to_cartesian_rotation",
]
