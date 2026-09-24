"""Exact invariant subspaces for primitive cluster stabilizers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlfcs.core.algebra.exact import certified_pivots, saturated_kernel


def label_symmetric_basis(labels: Sequence[object]) -> np.ndarray:
    """Return the 0/1 tensor basis symmetric between repeated lattice sites."""
    groups = []
    for label in dict.fromkeys(labels):
        positions = tuple(index for index, value in enumerate(labels) if value == label)
        if len(positions) > 1:
            groups.append(positions)
    order = len(labels)
    if not groups:
        return np.eye(3**order, dtype=np.int64)
    classes: dict[tuple[int, ...], list[int]] = {}
    for flat, component in enumerate(np.ndindex((3,) * order)):
        canonical = list(component)
        for positions in groups:
            directions = sorted(canonical[position] for position in positions)
            for position, direction in zip(positions, directions, strict=True):
                canonical[position] = direction
        classes.setdefault(tuple(canonical), []).append(flat)
    basis = np.zeros((3**order, len(classes)), dtype=np.int64)
    for column, members in enumerate(classes.values()):
        basis[members, column] = 1
    return basis


def apply_lattice_action(
    values: np.ndarray, rotation: np.ndarray, permutation: tuple[int, ...]
) -> np.ndarray:
    """Apply one tensor action after proving its int64 work-buffer bound."""
    order = len(permutation)
    exact_rotation = np.asarray(rotation)
    if not np.issubdtype(exact_rotation.dtype, np.integer):
        raise TypeError("lattice tensor actions require declared integer rotations")
    maximum_rotation = max((abs(int(value)) for value in exact_rotation.flat), default=0)
    bound = (3 * maximum_rotation) ** order
    if bound > np.iinfo(np.int64).max:
        raise OverflowError(f"the proven lattice tensor-action bound {bound} does not fit int64")
    tensors = values.T.reshape((-1,) + (3,) * order)
    transformed = tensors
    exact_rotation = np.asarray(exact_rotation, dtype=np.int64)
    for axis in range(order):
        transformed = np.tensordot(exact_rotation, transformed, axes=((1,), (axis + 1,)))
        transformed = np.moveaxis(transformed, 0, axis + 1)
    axes = (0,) + tuple(axis + 1 for axis in permutation)
    return np.transpose(transformed, axes).reshape(values.shape[1], -1).T


def invariant_basis(
    labels: Sequence[object],
    stabilizers: Sequence[tuple[np.ndarray, tuple[int, ...]]],
) -> np.ndarray:
    """Return a saturated Python-integer basis of the stabilizer invariant space."""
    label_basis = label_symmetric_basis(labels)
    blocks = [
        apply_lattice_action(label_basis, rotation, permutation) - label_basis
        for rotation, permutation in stabilizers
    ]
    constraints = (
        np.vstack(blocks) if blocks else np.empty((0, label_basis.shape[1]), dtype=np.int64)
    )
    rank, independent, _ = certified_pivots(constraints)
    dimension = label_basis.shape[1] - rank
    if dimension == 0:
        return np.empty((3 ** len(labels), 0), dtype=object)
    reduced = constraints[independent]
    kernel = saturated_kernel(reduced, expected_nullity=dimension)
    if kernel.shape[1] != dimension:
        raise RuntimeError(f"invariant kernel has {kernel.shape[1]} columns, expected {dimension}")
    result = label_basis.astype(object) @ kernel
    if not all(type(value) is int for value in result.flat):
        result = np.asarray([[int(value) for value in row] for row in result], dtype=object)
    return result


__all__ = ["apply_lattice_action", "invariant_basis", "label_symmetric_basis"]
