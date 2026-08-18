"""Shared translational equality constraints and their projection."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

from mlfcs.clusters.orbits import OrbitSpace


def build_translational_constraints(
    orbit_space: OrbitSpace,
    *,
    tolerance: float = 1e-12,
) -> sparse.csr_matrix:
    """Build the order-local acoustic sum-rule matrix."""
    dimensions = [orbit.dimension for orbit in orbit_space.orbits]
    offsets = np.cumsum([0, *dimensions])
    equations: dict[tuple[int, ...], int] = {}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for orbit_index, orbit in enumerate(orbit_space.orbits):
        representative_from_pivots = orbit.basis @ np.linalg.inv(orbit.basis[orbit.pivots])
        for image in orbit.images:
            image_from_pivots = image.action.apply_columns(representative_from_pivots)
            for component in range(3**orbit_space.order):
                directions = np.unravel_index(component, (3,) * orbit_space.order)
                key = image.cluster[:-1] + tuple(int(value) for value in directions)
                equation = equations.setdefault(key, len(equations))
                nonzero = np.flatnonzero(np.abs(image_from_pivots[component]) > tolerance)
                rows.extend([equation] * len(nonzero))
                columns.extend(int(offsets[orbit_index] + value) for value in nonzero)
                data.extend(float(image_from_pivots[component, value]) for value in nonzero)
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


__all__ = [
    "build_translational_constraints",
    "maximum_constraint_residual",
    "project_parameters",
]
