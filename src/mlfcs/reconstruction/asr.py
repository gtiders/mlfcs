from __future__ import annotations

from itertools import pairwise

import numpy as np
from ase import Atoms
from scipy import sparse

from mlfcs.core.constraints import (
    build_harmonic_rotational_constraints,
    build_translational_constraints,
    maximum_constraint_residual,
    project_parameters,
)
from mlfcs.core.orbits import OrbitSpace


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


def build_rotational_constraints(
    orbit_space: OrbitSpace,
    supercell: Atoms,
    *,
    tolerance: float = 1e-12,
) -> sparse.csr_matrix:
    """Compatibility wrapper for the shared FC1=0 Born--Huang boundary."""
    return build_harmonic_rotational_constraints(orbit_space, supercell, tolerance=tolerance)


def project_sum_rules(
    orbit_space: OrbitSpace,
    pivot_values: list[np.ndarray],
    *,
    supercell: Atoms,
    acoustic: bool,
    rotational: bool,
    tolerance: float = 1e-9,
) -> tuple[list[np.ndarray], dict[str, tuple[float, float]]]:
    """Project onto selected translational and rotational constraint spaces."""
    translational = build_translational_constraints(orbit_space)
    matrices = {"translational": translational}
    selected: list[sparse.csr_matrix] = []
    if acoustic:
        selected.append(translational)
    if rotational:
        rotational_matrix = build_rotational_constraints(orbit_space, supercell)
        matrices["rotational"] = rotational_matrix
        selected.append(rotational_matrix)

    offsets = np.cumsum([0] + [len(values) for values in pivot_values])
    original = np.concatenate(pivot_values)
    constraints = sparse.vstack(selected, format="csr") if selected else None
    projected = (
        original.copy()
        if constraints is None
        else project_parameters(constraints, original, tolerance=tolerance)
    )
    drifts = {
        name: (
            maximum_constraint_residual(matrix, original),
            maximum_constraint_residual(matrix, projected),
        )
        for name, matrix in matrices.items()
    }
    values = [projected[begin:end] for begin, end in pairwise(offsets)]
    return values, drifts


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
