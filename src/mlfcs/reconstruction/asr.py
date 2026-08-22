from __future__ import annotations

from itertools import pairwise

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy import sparse
from scipy.sparse.linalg import lsmr

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
    """Build harmonic Born-Huang rotational constraints in pivot space.

    For every force component and rotation axis this imposes zero force under
    the infinitesimal rigid displacement ``u_j = omega x r_ij``. Relative MIC
    vectors make the equations independent of the coordinate origin.
    """
    if orbit_space.order != 2:
        raise ValueError(
            "rotational sum rules are currently available only for order 2; "
            "higher-order conditions couple adjacent force-constant orders"
        )
    dimensions = [orbit.dimension for orbit in orbit_space.orbits]
    offsets = np.cumsum([0, *dimensions])
    equations: dict[tuple[int, int, int], int] = {}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    axes = np.eye(3)

    for orbit_index, orbit in enumerate(orbit_space.orbits):
        representative_from_pivots = orbit.basis @ np.linalg.inv(orbit.basis[orbit.pivots])
        for image in orbit.images:
            first, second = image.cluster
            vector, _ = find_mic(
                supercell.positions[second] - supercell.positions[first],
                supercell.cell,
                supercell.pbc,
            )
            rigid_displacements = np.cross(axes, vector)
            image_from_pivots = image.action.apply_columns(representative_from_pivots)
            for force_direction in range(3):
                block = image_from_pivots.reshape(3, 3, -1)[force_direction]
                for rotation_axis in range(3):
                    equation_key = (int(first), force_direction, rotation_axis)
                    equation = equations.setdefault(equation_key, len(equations))
                    coefficients = rigid_displacements[rotation_axis] @ block
                    nonzero = np.flatnonzero(np.abs(coefficients) > tolerance)
                    rows.extend([equation] * len(nonzero))
                    columns.extend(int(offsets[orbit_index] + value) for value in nonzero)
                    data.extend(float(coefficients[value]) for value in nonzero)

    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(equations), int(offsets[-1])),
    ).tocsr()


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
        else _project_parameters(constraints, original, tolerance=tolerance)
    )
    drifts = {
        name: (
            _maximum_residual(matrix, original),
            _maximum_residual(matrix, projected),
        )
        for name, matrix in matrices.items()
    }
    values = [projected[begin:end] for begin, end in pairwise(offsets)]
    return values, drifts


def build_translational_constraints(
    orbit_space: OrbitSpace,
    *,
    tolerance: float = 1e-12,
) -> sparse.csr_matrix:
    """Build the order-local acoustic sum-rule matrix ``A``.

    Each row fixes the first ``n - 1`` atom indices and every Cartesian
    component, and sums only the final atom index. Permutation symmetry in
    the orbit space makes the equivalent constraints on other axes redundant.
    """
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
                equation_key = image.cluster[:-1] + tuple(int(x) for x in directions)
                equation = equations.setdefault(equation_key, len(equations))
                nonzero = np.flatnonzero(np.abs(image_from_pivots[component]) > tolerance)
                rows.extend([equation] * len(nonzero))
                columns.extend(int(offsets[orbit_index] + value) for value in nonzero)
                data.extend(float(image_from_pivots[component, value]) for value in nonzero)

    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(equations), int(offsets[-1])),
    ).tocsr()


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
    parameters = _project_parameters(constraints, parameters, tolerance=tolerance)
    final_residual = constraints @ parameters
    projected = [parameters[begin:end] for begin, end in pairwise(offsets)]
    final_drift = float(np.linalg.norm(final_residual, ord=np.inf))
    return (projected, initial_drift, final_drift) if return_drift else projected


def _project_parameters(
    constraints: sparse.csr_matrix,
    parameters: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    scale = max(float(np.linalg.norm(parameters)), 1.0)
    # LSMR returns the minimum-norm correction solving A @ correction =
    # A @ parameters. Subtracting it is the orthogonal projection onto
    # null(A), without forming a dense Gram matrix or squaring its condition
    # number. Repeating the projection only refines finite-precision residuals.
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
        parameters = parameters - correction

    final_residual = constraints @ parameters
    if float(np.linalg.norm(final_residual, ord=np.inf)) > tolerance * scale:
        raise RuntimeError(
            "sum-rule projection did not converge: "
            f"max residual={np.linalg.norm(final_residual, ord=np.inf):.3e}"
        )
    return parameters


def _maximum_residual(constraints: sparse.csr_matrix, parameters: np.ndarray) -> float:
    if constraints.shape[0] == 0 or constraints.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm(constraints @ parameters, ord=np.inf))
