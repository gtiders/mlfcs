"""One compiled integer kernel for the complete action on one cluster."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def transform_cluster(
    labels: np.ndarray,
    rotations: np.ndarray,
    site_permutations: np.ndarray,
    site_shifts: np.ndarray,
    axis_permutations: np.ndarray,
) -> np.ndarray:
    """Apply every space/axis operation and re-anchor each result."""
    operation_count = rotations.shape[0]
    permutation_count = axis_permutations.shape[0]
    order = labels.shape[0]
    transformed = np.empty((operation_count * permutation_count, order, 4), dtype=np.int64)
    for operation in range(operation_count):
        for permutation_index in range(permutation_count):
            target = operation * permutation_count + permutation_index
            for axis in range(order):
                source_axis = axis_permutations[permutation_index, axis]
                source_site = labels[source_axis, 0]
                transformed[target, axis, 0] = site_permutations[operation, source_site]
                for component in range(3):
                    value = site_shifts[operation, source_site, component]
                    for inner in range(3):
                        value += (
                            labels[source_axis, inner + 1] * rotations[operation, component, inner]
                        )
                    transformed[target, axis, component + 1] = value
            for component in range(3):
                origin = transformed[target, 0, component + 1]
                for axis in range(order):
                    transformed[target, axis, component + 1] -= origin
    return transformed


__all__ = ["transform_cluster"]
