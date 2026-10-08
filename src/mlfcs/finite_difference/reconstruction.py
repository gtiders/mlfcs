"""Recover primitive force constants from ordered finite-difference samples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

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
    physical = np.concatenate(coefficients)
    coordinates = space.acoustic_coordinates(order)
    if coordinates is not None:
        # Project the reconstructed physical parameters onto the acoustic subspace.
        if coordinates.dimension:
            operator = LinearOperator(
                (coordinates.width, coordinates.dimension),
                matvec=coordinates.lift,
                rmatvec=coordinates.adjoint,
            )
            solution = lsmr(
                operator,
                physical,
                atol=1e-12,
                btol=1e-12,
                maxiter=max(1000, 2 * coordinates.dimension),
            )
            if solution[1] not in (0, 1, 2, 4, 5):
                raise RuntimeError("acoustic constraint projection did not converge")
            physical = coordinates.lift(solution[0])
        else:
            # A zero-dimensional subspace contains only the zero parameter vector.
            physical = np.zeros(coordinates.width)
    return ForceConstants(space, {order: physical})


__all__ = ["reconstruct_force_constants"]
