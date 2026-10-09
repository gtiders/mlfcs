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
    """Recover one primitive force-constant order from finite-difference forces.

    Dataset frames must follow the canonical sampling order defined by
    ``finite_difference``. This ordering is assumed and is not inferred from
    frame geometry or metadata.

    For an order-``p`` force constant, mixed central differences of the forces
    provide the required ``p - 1`` displacement derivatives. When multiple
    step lengths are supplied, these derivatives are extrapolated to zero step
    through the even central-difference error expansion.

    Representative Cartesian tensor components are converted to the physical
    parameter coordinates of the primitive symmetry orbits. If acoustic
    coordinates are available for this order, the reconstructed parameter
    vector is projected onto that constrained subspace by least squares.

    The returned ``ForceConstants`` contains only the selected order.
    """
    if not isinstance(dataset, ForceDataset):
        raise TypeError("reap() requires a ForceDataset")
    expected = (finite_difference.n_configurations, finite_difference.cluster_map.n_atoms, 3)
    forces = dataset.forces
    if forces.shape != expected or not np.all(np.isfinite(forces)):
        raise ValueError(f"finite-difference forces must be finite with shape {expected}")
    order = finite_difference.order
    sign_combinations = np.asarray(finite_difference._sign_combinations, dtype=np.float64)
    # The sign product forms the mixed central stencil; the leading minus maps
    # force derivatives to energy derivatives.
    sign_weights = np.prod(sign_combinations, axis=1)
    step_weights = _extrapolation_weights(finite_difference.disps)
    sign_count = len(sign_combinations)
    step_count = len(finite_difference.disps)
    force_constant_estimates: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for key_index, key in enumerate(finite_difference._displacement_keys):
        estimates = []
        base = key_index * step_count * sign_count
        for step_index, step_length in enumerate(finite_difference.disps):
            begin = base + step_index * sign_count
            estimates.append(
                -np.tensordot(
                    sign_weights,
                    forces[begin : begin + sign_count],
                    axes=(0, 0),
                )
                / (2.0 * step_length) ** (order - 1)
            )
        force_constant_estimates[key] = np.tensordot(step_weights, estimates, axes=(0, 0))

    cluster_map = finite_difference.cluster_map
    space = cluster_map.cluster_space
    block = space.block(order)
    orbit_parameters = []
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        orbit = space.orbits[orbit_index]
        representative_image = orbit.clusters.index(orbit.representative)
        atom_indices = tuple(
            int(value)
            for value in cluster_map.image_atom_indices[orbit_index][representative_image]
        )
        observation_components = []
        for row in orbit.observation_rows:
            directions = np.unravel_index(int(row), (3,) * order)
            key = tuple((atom_indices[axis], int(directions[axis])) for axis in range(order - 1))
            observation_components.append(
                force_constant_estimates[key][atom_indices[-1], int(directions[-1])]
            )
        orbit_parameters.append(
            np.linalg.solve(
                orbit.observation_matrix, np.asarray(observation_components, dtype=np.float64)
            )
        )
    physical_parameters = np.concatenate(orbit_parameters)
    acoustic_coordinates = space.acoustic_coordinates(order)
    if acoustic_coordinates is not None:
        # Project the reconstructed physical parameters onto the acoustic subspace.
        physical_parameters = acoustic_coordinates.project(physical_parameters)
    return ForceConstants(space, {order: physical_parameters})


__all__ = ["reconstruct_force_constants"]
