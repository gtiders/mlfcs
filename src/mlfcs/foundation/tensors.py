"""Shared Cartesian tensor dimensions and symmetry actions."""

import numpy as np
from numba import njit

from mlfcs.foundation.arrays import require_bound


def tensor_dimension(order):
    """Return the number of Cartesian components of an order-p tensor.

    Each tensor index has three Cartesian directions, so the flattened
    dimension is 3**order.

    Raises
    ------
    OverflowError
        If the component dimension exceeds the supported integer domain.
    """
    dimension = 1
    for _ in range(order):
        dimension = require_bound("tensor component dimension", dimension * 3)
    return dimension


def tensor_action_bound(rotation, order, coefficient=1):
    """Return a conservative coefficient bound for an integer tensor action.

    The bound is coefficient * max(1, ||rotation||_inf)**order, where
    ||rotation||_inf is the maximum absolute row sum.
    """
    norm = max(sum(abs(int(v)) for v in row) for row in rotation)
    return int(coefficient) * max(1, norm) ** order


def validate_tensor_action(rotation, order, coefficient=1, subtraction=0):
    """Require an integer tensor action to remain within the supported range.

    coefficient bounds the input magnitude and subtraction reserves
    additional range for a subsequent difference operation.

    Returns
    -------
    int
        The validated worst-case magnitude.
    """
    bound = tensor_action_bound(rotation, order, coefficient) + subtraction
    return require_bound("tensor action intermediate", bound)


@njit(cache=True)
def apply_tensor_action(values, rotation, permutation):
    """Apply an integer-valued symmetry action to flattened tensor basis vectors.

    Each column represents an order-p tensor flattened in C order. The integer
    matrix acts on every tensor index, then permutation reorders the tensor axes
    using NumPy transpose convention. Integer callers must validate the
    corresponding arithmetic bound before calling this kernel.
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
    """Rotate every Cartesian tensor index and then permute the tensor axes.

    Rotation acts independently on each index in the Cartesian basis.
    Permutation gives the final index order following NumPy transpose
    convention. The input tensor is not modified.
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
    """Rotate and permute every tensor represented by the basis columns.

    Each column is an order-p Cartesian tensor flattened in C order. The same
    action defined by rotate_tensor is applied independently to every column.
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
