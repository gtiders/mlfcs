"""Shared translational equality constraints and their projection."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr


def build_translational_constraints(
    orbit_space,
    *,
    tolerance: float = 1e-12,
) -> sparse.csr_matrix:
    """Build the order-local acoustic sum-rule matrix.

    The parameters are the coefficients of each orbit's Cartesian basis, so a row is
    assembled from ``action.apply_columns(cartesian_basis)``: the component of an image
    tensor is a linear combination of the orbit parameters.  A coefficient is dropped
    only below ``tolerance``, which separates the floating point noise of an
    algebraically zero component from genuine coefficients by orders of magnitude;
    dropping the noise is what keeps the rows of one physical constraint identical, and
    the fitting stage then de-duplicates and ranks them.
    """
    dimensions = [orbit.dimension for orbit in orbit_space.orbits]
    offsets = np.cumsum([0, *dimensions])
    equations: dict[tuple[int, ...], int] = {}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for orbit_index, orbit in enumerate(orbit_space.orbits):
        cartesian_basis = np.asarray(orbit.cartesian_basis, dtype=float)
        for image in orbit.images:
            image_from_parameters = image.action.apply_columns(cartesian_basis)
            for component in range(3**orbit_space.order):
                directions = np.unravel_index(component, (3,) * orbit_space.order)
                labels = image.key.labels if hasattr(image, "key") else image.cluster
                key = tuple(labels[:-1]) + tuple(int(value) for value in directions)
                equation = equations.setdefault(key, len(equations))
                nonzero = np.flatnonzero(np.abs(image_from_parameters[component]) > tolerance)
                rows.extend([equation] * len(nonzero))
                columns.extend(int(offsets[orbit_index] + value) for value in nonzero)
                data.extend(float(image_from_parameters[component, value]) for value in nonzero)
    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(equations), int(offsets[-1])),
    ).tocsr()


def project_parameters(
    constraints: sparse.csr_matrix,
    parameters: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Orthogonally project parameters onto ``null(constraints)`` with LSMR."""
    parameters = np.asarray(parameters).copy()
    if constraints.shape[0] == 0 or constraints.shape[1] == 0:
        return parameters
    scale = max(float(np.linalg.norm(parameters)), 1.0)
    for _ in range(8):
        residual = constraints @ parameters
        if float(np.linalg.norm(residual, ord=np.inf)) <= tolerance * scale:
            break
        correction = lsmr(
            constraints,
            residual,
            atol=tolerance * 0.01,
            btol=tolerance * 0.01,
            maxiter=max(1000, 4 * constraints.shape[1]),
        )[0]
        parameters -= correction
    final = maximum_constraint_residual(constraints, parameters)
    if final > tolerance * scale:
        raise RuntimeError(f"constraint projection did not converge: max residual={final:.3e}")
    return parameters


def maximum_constraint_residual(
    constraints: sparse.csr_matrix,
    parameters: np.ndarray,
) -> float:
    if constraints.shape[0] == 0 or constraints.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm(constraints @ parameters, ord=np.inf))


def maximum_acoustic_sum_rule_drift(orbit_space, coefficients: list[np.ndarray]) -> float:
    """Return the largest atomic-sum residual of one orbit parameter vector."""
    constraints = build_translational_constraints(orbit_space)
    if constraints.shape[0] == 0 or constraints.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm(constraints @ np.concatenate(coefficients), ord=np.inf))


def project_acoustic_sum_rule(
    orbit_space,
    coefficients: list[np.ndarray],
    *,
    tolerance: float = 1e-9,
    return_drift: bool = False,
):
    """Project orbit coefficients onto the translational null space."""
    offsets = np.cumsum([0] + [len(values) for values in coefficients])
    parameters = np.concatenate(coefficients)
    constraints = build_translational_constraints(orbit_space)
    if constraints.shape[0] == 0 or constraints.shape[1] == 0:
        return (coefficients, 0.0, 0.0) if return_drift else coefficients
    initial_drift = float(np.linalg.norm(constraints @ parameters, ord=np.inf))
    parameters = project_parameters(constraints, parameters, tolerance=tolerance)
    projected = [parameters[begin:end] for begin, end in pairwise(offsets)]
    final_drift = float(np.linalg.norm(constraints @ parameters, ord=np.inf))
    return (projected, initial_drift, final_drift) if return_drift else projected


__all__ = [
    "build_translational_constraints",
    "maximum_acoustic_sum_rule_drift",
    "maximum_constraint_residual",
    "project_acoustic_sum_rule",
    "project_parameters",
]
