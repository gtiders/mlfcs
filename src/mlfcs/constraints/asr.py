from __future__ import annotations

from itertools import pairwise

import numpy as np

from mlfcs.clusters.orbits import OrbitSpace
from mlfcs.constraints.translational import (
    build_translational_constraints,
    project_parameters,
)


def maximum_acoustic_sum_rule_drift(
    orbit_space: OrbitSpace,
    pivot_values: list[np.ndarray],
) -> float:
    """Return the largest atomic-sum residual, analogous to phonopy's drift."""
    constraints = build_translational_constraints(orbit_space)
    if constraints.shape[0] == 0 or constraints.shape[1] == 0:
        return 0.0
    residual = constraints @ np.concatenate(pivot_values)
    return float(np.linalg.norm(residual, ord=np.inf))


def project_acoustic_sum_rule(
    orbit_space: OrbitSpace,
    pivot_values: list[np.ndarray],
    *,
    tolerance: float = 1e-9,
    return_drift: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], float, float]:
    """Orthogonally project independent IFC parameters onto ``null(A)``."""
    offsets = np.cumsum([0] + [len(values) for values in pivot_values])
    parameters = np.concatenate(pivot_values)
    constraints = build_translational_constraints(orbit_space)
    if constraints.shape[0] == 0 or constraints.shape[1] == 0:
        return (pivot_values, 0.0, 0.0) if return_drift else pivot_values

    initial_drift = float(np.linalg.norm(constraints @ parameters, ord=np.inf))
    parameters = project_parameters(constraints, parameters, tolerance=tolerance)
    final_residual = constraints @ parameters
    projected = [parameters[begin:end] for begin, end in pairwise(offsets)]
    final_drift = float(np.linalg.norm(final_residual, ord=np.inf))
    return (projected, initial_drift, final_drift) if return_drift else projected
