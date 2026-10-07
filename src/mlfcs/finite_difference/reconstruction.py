"""Recover primitive force constants from ordered finite-difference samples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mlfcs.dataset import ForceDataset
from mlfcs.force_constants import ForceConstants

if TYPE_CHECKING:
    from mlfcs.finite_difference.sampling import FiniteDifference


def _extrapolation_weights(disps: tuple[float, ...]) -> np.ndarray:
    """Return weights that extrapolate the even central-difference error to zero step."""
    squared = np.square(np.asarray(disps, dtype=np.float64))
    weights = np.ones(len(squared), dtype=np.float64)
    for index, value in enumerate(squared):
        for other, other_value in enumerate(squared):
            if other != index:
                weights[index] *= -other_value / (value - other_value)
    return weights


def reconstruct_force_constants(
    finite_difference: FiniteDifference,
    dataset: ForceDataset,
) -> ForceConstants:
    """Recover one force-constant order from its canonical force samples.

    Dataset frames must follow the sampling plan's canonical displacement order.
    Frame geometry and metadata are not used to verify that order. Central mixed derivatives are extrapolated across
    the requested step lengths, then mapped through each orbit's observation
    matrix. The result uses the primitive cluster-space parameterization and
    contains only the selected order.
    """
    if not isinstance(dataset, ForceDataset):
        raise TypeError("reconstruct() requires a ForceDataset")
    expected = (finite_difference.n_configurations, finite_difference.cluster_map.n_atoms, 3)
    values = dataset.forces
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError(f"finite-difference forces must be finite with shape {expected}")
    order = finite_difference.order
    signs = np.asarray(finite_difference._signs, dtype=np.float64)
    # The sign product forms the mixed central stencil; the leading minus maps
    # force derivatives to energy derivatives.
    sign_weights = np.prod(signs, axis=1)
    disp_weights = _extrapolation_weights(finite_difference.disps)
    sign_count = len(signs)
    disp_count = len(finite_difference.disps)
    derivatives: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for key_index, key in enumerate(finite_difference._keys):
        estimates = []
        base = key_index * disp_count * sign_count
        for disp_index, disp in enumerate(finite_difference.disps):
            begin = base + disp_index * sign_count
            estimates.append(
                -np.tensordot(
                    sign_weights,
                    values[begin : begin + sign_count],
                    axes=(0, 0),
                )
                / (2.0 * disp) ** (order - 1)
            )
        derivatives[key] = np.tensordot(disp_weights, estimates, axes=(0, 0))

    cluster_map = finite_difference.cluster_map
    space = cluster_map.cluster_space
    block = space.block(order)
    coefficients = []
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        orbit = space.orbits[orbit_index]
        image = orbit.clusters.index(orbit.representative)
        atoms = tuple(int(value) for value in cluster_map.image_atom_indices[orbit_index][image])
        observed = []
        for row in orbit.observation_rows:
            directions = np.unravel_index(int(row), (3,) * order)
            key = tuple((atoms[axis], int(directions[axis])) for axis in range(order - 1))
            observed.append(derivatives[key][atoms[-1], int(directions[-1])])
        coefficients.append(
            np.linalg.solve(orbit.observation_matrix, np.asarray(observed, dtype=np.float64))
        )
    return ForceConstants(space, {order: np.concatenate(coefficients)})


__all__ = ["reconstruct_force_constants"]
