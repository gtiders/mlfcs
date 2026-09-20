"""Shared translational equality constraints and their projection."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

from mlfcs.interactions.algebra.actions import scaled_to_cartesian_matrix


def build_translational_constraints(
    orbit_space,
    *,
    tolerance: float = 1e-12,
) -> sparse.csr_matrix:
    """Build the order-local acoustic sum-rule matrix.

    A coefficient is dropped only below ``tolerance``, which separates the floating point
    noise of an algebraically zero component (measured at $3.5\times10^{-15}$ for a row
    scale of $59$) from genuine coefficients by orders of magnitude.  Dropping the noise is
    what keeps the rows of one physical constraint identical, which the fitting stage then
    de-duplicates and ranks, and that de-duplication normalizes the rows so the rank tests
    are scale free.
    """
    order = orbit_space.order
    dimensions = [orbit.dimension for orbit in orbit_space.orbits]
    offsets = np.cumsum([0, *dimensions])
    frame = scaled_to_cartesian_matrix(orbit_space.cell, order)
    equations: dict[tuple[int, ...], int] = {}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for orbit_index, orbit in enumerate(orbit_space.orbits):
        cartesian_basis = frame @ np.asarray(orbit.basis, dtype=np.int64)
        for image in orbit.images:
            image_from_coefficients = image.action.apply_columns(cartesian_basis)
            for component in range(3**order):
                directions = np.unravel_index(component, (3,) * order)
                labels = image.key.labels if hasattr(image, "key") else image.cluster
                key = tuple(labels[:-1]) + tuple(int(value) for value in directions)
                equation = equations.setdefault(key, len(equations))
                column_values = image_from_coefficients[component]
                nonzero = np.flatnonzero(np.abs(column_values) > tolerance)
                rows.extend([equation] * len(nonzero))
                columns.extend(int(offsets[orbit_index] + value) for value in nonzero)
                data.extend(float(image_from_coefficients[component, value]) for value in nonzero)
    return sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(equations), int(offsets[-1]))
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


def maximum_acoustic_sum_rule_drift(orbit_space, pivot_values: list[np.ndarray]) -> float:
    """Return the largest atomic-sum residual."""
    constraints = build_translational_constraints(orbit_space)
    if constraints.shape[0] == 0 or constraints.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm(constraints @ np.concatenate(pivot_values), ord=np.inf))


def project_acoustic_sum_rule(
    orbit_space,
    pivot_values: list[np.ndarray],
    *,
    tolerance: float = 1e-9,
    return_drift: bool = False,
):
    """Project independent IFC parameters onto the translational null space."""
    offsets = np.cumsum([0] + [len(values) for values in pivot_values])
    parameters = np.concatenate(pivot_values)
    constraints = build_translational_constraints(orbit_space)
    if constraints.shape[0] == 0 or constraints.shape[1] == 0:
        return (pivot_values, 0.0, 0.0) if return_drift else pivot_values
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
