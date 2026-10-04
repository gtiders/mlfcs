"""Shared lattice and Cartesian tensor transformations."""

import numpy as np
from numba import njit

from mlfcs._arrays import require_bound


def tensor_dimension(order):
    """Compute the actual tensor extent without constructing enormous powers."""
    dimension = 1
    for _ in range(order):
        dimension = require_bound("tensor component dimension", dimension * 3)
    return dimension


def tensor_action_bound(rotation, order, coefficient=1):
    """Bound every axis multiplication and partial sum, using actual rotation."""
    norm = max(sum(abs(int(v)) for v in row) for row in rotation)
    return int(coefficient) * max(1, norm) ** order


def prove_tensor_action(rotation, order, coefficient=1, subtraction=0):
    """Admit tensor-action products, partial sums and an optional subtraction in int64.

    ``rotation`` is the actual integer 3x3 action, ``order`` the tensor rank,
    and ``coefficient`` bounds absolute seed entries. The returned bound includes
    ``subtraction``; OverflowError is raised before unchecked tensor operations.
    """
    bound = tensor_action_bound(rotation, order, coefficient) + subtraction
    return require_bound("tensor action intermediate", bound)


@njit(cache=True)
def _tensor_action(values, rotation, permutation):
    """Apply axis-wise rotations then permute flattened tensor rows.

    ``values`` has shape (3**p, n_columns), ``rotation`` (3, 3), and
    ``permutation`` (p,). Rows encode Cartesian directions in C order, with the
    last axis varying fastest. Inputs are preserved; output has the seed dtype.
    Integer callers must admit tensor extent and prove_tensor_action first.
    """
    order = len(permutation)
    dimension, columns = values.shape
    transformed = values.copy()
    stride = dimension
    for axis in range(order):
        stride //= 3
        output = np.zeros_like(values)
        for row in range(dimension):
            direction = (row // stride) % 3
            base = row - direction * stride
            for inner in range(3):
                for column in range(columns):
                    output[row, column] += (
                        rotation[direction, inner] * transformed[base + inner * stride, column]
                    )
        transformed = output
    result = np.empty_like(values)
    for row in range(dimension):
        source = 0
        for axis in range(order):
            # np.transpose(..., permutation) reads source axis permutation[axis].
            direction = row // (3 ** (order - 1 - axis)) % 3
            source += direction * (3 ** (order - 1 - permutation[axis]))
        result[row] = transformed[source]
    return result


def rotate_tensor(
    tensor: np.ndarray,
    rotation: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """Return a Cartesian tensor after rotation on every axis and axis permutation.

    ``tensor`` has shape (3,) repeated p times; ``rotation`` is a 3x3 Cartesian
    matrix and ``permutation`` contains p axis indices. Inputs are not modified.
    The transformation contracts rotation's second index with each tensor axis,
    then applies NumPy's transpose convention.
    """
    result = tensor
    for axis in range(tensor.ndim):
        result = np.tensordot(rotation, result, axes=((1,), (axis,)))
        result = np.moveaxis(result, 0, axis)
    return np.transpose(result, tuple(int(value) for value in permutation))


def rotate_basis(
    basis: np.ndarray,
    rotation: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """Transform every Cartesian basis column and return a float64 matrix.

    ``basis`` has shape (3**p, dimension), flattened in C order. Rotation and
    permutation follow rotate_tensor; the input basis remains unchanged.
    """
    order = len(permutation)
    result = np.empty_like(basis, dtype=np.float64)
    for column in range(basis.shape[1]):
        result[:, column] = rotate_tensor(
            basis[:, column].reshape((3,) * order),
            rotation,
            permutation,
        ).reshape(-1)
    return result
