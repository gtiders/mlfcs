"""Shared equality-constraint construction, diagnostics, and projection."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy import sparse
from scipy.sparse.linalg import lsmr

from mlfcs.core.geometry import PeriodicGeometry
from mlfcs.core.orbits import OrbitSpace


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


def build_harmonic_rotational_constraints(
    orbit_space: OrbitSpace,
    supercell: Atoms,
    *,
    tolerance: float = 1e-12,
) -> sparse.csr_matrix:
    """Build the FC1=0 Born--Huang rotational boundary for harmonic IFCs.

    This is the lowest member of the adjacent-order rotational hierarchy.  At
    a mechanical-equilibrium reference structure FC1 vanishes, leaving the
    condition that a rigid infinitesimal rotation produces no harmonic force.
    Relative MIC vectors make the equations independent of coordinate origin.
    """
    if orbit_space.order != 2:
        raise ValueError("the FC1=0 rotational boundary requires order-2 force constants")
    dimensions = [orbit.dimension for orbit in orbit_space.orbits]
    offsets = np.cumsum([0, *dimensions])
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    axes = np.eye(3)
    geometry = PeriodicGeometry(supercell.cell, supercell.pbc)

    for orbit_index, orbit in enumerate(orbit_space.orbits):
        representative_from_pivots = orbit.basis @ np.linalg.inv(orbit.basis[orbit.pivots])
        for image in orbit.images:
            first, second = image.cluster
            vector, _ = geometry.mic(supercell.positions[second] - supercell.positions[first])
            rigid_displacements = np.cross(axes, vector)
            image_from_pivots = image.action.apply_columns(representative_from_pivots)
            for force_direction in range(3):
                block = image_from_pivots.reshape(3, 3, -1)[force_direction]
                for rotation_axis in range(3):
                    equation = (int(first) * 3 + force_direction) * 3 + rotation_axis
                    coefficients = rigid_displacements[rotation_axis] @ block
                    nonzero = np.flatnonzero(np.abs(coefficients) > tolerance)
                    rows.extend([equation] * len(nonzero))
                    columns.extend(int(offsets[orbit_index] + value) for value in nonzero)
                    data.extend(float(coefficients[value]) for value in nonzero)

    n_anchors = 1 + max(
        (int(image.cluster[0]) for orbit in orbit_space.orbits for image in orbit.images),
        default=-1,
    )
    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(9 * n_anchors, int(offsets[-1])),
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
    "build_harmonic_rotational_constraints",
    "build_translational_constraints",
    "maximum_constraint_residual",
    "project_parameters",
]
