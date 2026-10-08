"""FC2 rotational-invariance and zero-stress equilibrium projection.

Born-Huang conditions constrain first moments of force constants and
interatomic separations. Huang conditions constrain second moments at zero
stress. Corrections preserve the existing acoustic residual and minimize
change in physical component coordinates within the resolvable subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse

from mlfcs.force_constants.model import ForceConstants
from mlfcs.foundation.log import get_logger
from mlfcs.foundation.tensors import rotate_basis

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RotationResult:
    """FC2 rotational correction, physical residuals and resolved constraint rank.

    ``force_constants`` contains the corrected model; higher orders retain
    their input coefficients. The correction preserves the initial acoustic
    residual rather than applying the acoustic sum rule.

    ``length_scale`` is the median non-onsite pair distance in angstrom.
    Absolute Born-Huang and Huang residuals have units of
    eV/angstrom and eV, respectively. Disabled rotational
    conditions report ``None``. Relative residuals use moment equations
    divided by the corresponding power of ``length_scale``, normalized by
    the largest row absolute sum times the largest parameter magnitude.

    ``correction_norm`` is the Euclidean change in physical FC2 parameters;
    ``relative_correction`` divides it by their initial norm. ``retained_rank``
    counts effective singular directions above ``rank_cutoff``. The adjacent
    retained/discarded singular values are ``None`` when the corresponding
    set is empty. ``automatic_rank`` indicates whether ``rank_rtol`` was
    estimated from geometry rather than supplied by the caller.

    ``geometry_residual`` is the largest symmetry-related position mismatch
    in angstrom; ``orthogonality_residual`` measures the dimensionless
    departure of Cartesian symmetry operations from orthogonality.
    """

    force_constants: ForceConstants
    born_huang: bool
    huang: bool
    length_scale: float
    equations: int
    born_huang_before: float | None
    born_huang_after: float | None
    huang_before: float | None
    huang_after: float | None
    relative_before: float
    relative_after: float
    correction_norm: float
    relative_correction: float
    retained_rank: int
    rank_rtol: float
    automatic_rank: bool
    rank_cutoff: float
    smallest_retained_singular_value: float | None
    largest_discarded_singular_value: float | None
    geometry_residual: float
    orthogonality_residual: float


def _append(
    entries: tuple[list[int], list[int], list[float]],
    row: int,
    column_start: int,
    values: np.ndarray,
) -> None:
    """Collect nonzero parameter contributions to one rotational moment equation."""
    rows, columns, data = entries
    for column, value in enumerate(values):
        if value != 0.0:
            rows.append(row)
            columns.append(column_start + column)
            data.append(float(value))


def _fc2_moment_matrices(
    cluster_space,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, float]:
    """Construct Born-Huang first-moment and Huang second-moment constraints.

    Use Cartesian separations including periodic images, divided by the
    median non-onsite pair distance in angstrom. Born-Huang equations are
    antisymmetric in their displacement and moment directions; Huang
    equations compare exchanged tensor/moment axis pairs after summing over
    primitive atoms. Columns follow the physical FC2 parameters.
    No positive finite non-onsite length scale raises ValueError.
    """
    space = cluster_space
    block = space.block(2)
    parameter_count = block.parameters.stop - block.parameters.start
    offsets = np.cumsum(
        [
            0,
            *(
                space.orbits[index].dimension
                for index in range(block.orbits.start, block.orbits.stop)
            ),
        ]
    )
    images = []
    lengths = []
    positions = space.cartesian_positions
    cell = space.cell
    for local_orbit, orbit_index in enumerate(range(block.orbits.start, block.orbits.stop)):
        orbit = space.orbits[orbit_index]
        for image, cluster in enumerate(orbit.clusters):
            first, second = cluster.sites
            vector = (
                positions[second.site]
                - positions[first.site]
                + np.asarray(second.translation, dtype=np.float64) @ cell
            )
            if second != first:
                lengths.append(float(np.linalg.norm(vector)))
            rotation = space.symmetry.cartesian_rotations[orbit.operations[image]].T
            basis = rotate_basis(orbit.component_basis, rotation, orbit.permutations[image])
            images.append((first.site, vector, int(offsets[local_orbit]), basis))
    if not lengths:
        raise ValueError("rotational invariance requires at least one non-onsite FC2 pair")
    length_scale = float(np.median(lengths))
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("non-onsite FC2 pairs do not define a positive finite length scale")

    born_entries = ([], [], [])
    huang_entries = ([], [], [])
    for first, vector, column_start, basis in images:
        reduced = vector / length_scale
        for alpha in range(3):
            for pair, (beta, gamma) in enumerate(((0, 1), (0, 2), (1, 2))):
                row = (first * 3 + alpha) * 3 + pair
                values = reduced[gamma] * basis[3 * alpha + beta]
                values = values - reduced[beta] * basis[3 * alpha + gamma]
                _append(born_entries, row, column_start, values)
        dyadic = np.outer(reduced, reduced)
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    for delta in range(3):
                        row = (alpha * 3 + beta) * 9 + gamma * 3 + delta
                        values = dyadic[gamma, delta] * basis[3 * alpha + beta]
                        values = values - dyadic[alpha, beta] * basis[3 * gamma + delta]
                        _append(huang_entries, row, column_start, values)

    born = sparse.coo_matrix(
        (born_entries[2], (born_entries[0], born_entries[1])),
        shape=(space.n_atoms * 9, parameter_count),
    ).tocsr()
    huang = sparse.coo_matrix(
        (huang_entries[2], (huang_entries[0], huang_entries[1])),
        shape=(81, parameter_count),
    ).tocsr()
    for matrix in (born, huang):
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
    return born, huang, length_scale


def _relative_residual(matrix: sparse.csr_matrix, values: np.ndarray) -> tuple[float, float]:
    """Measure rotational violation relative to row norm and parameter magnitude.

    Return the largest absolute residual and its ratio to the largest row
    absolute sum times the largest parameter magnitude. Empty systems and
    zero scales return zero.
    """
    if matrix.shape[0] == 0 or values.size == 0:
        return 0.0, 0.0
    residual = np.asarray(matrix @ values)
    maximum = float(np.max(np.abs(residual), initial=0.0))
    row_norms = np.asarray(np.abs(matrix).sum(axis=1)).reshape(-1)
    equation_scale = float(np.max(row_norms, initial=0.0))
    parameter_scale = float(np.max(np.abs(values), initial=0.0))
    scale = equation_scale * parameter_scale
    relative = maximum / scale if scale else 0.0
    return maximum, relative


def _maximum_residual(matrix: sparse.csr_matrix, values: np.ndarray) -> float:
    """Return the largest absolute moment violation, or zero for no equations."""
    if matrix.shape[0] == 0:
        return 0.0
    return float(np.max(np.abs(matrix @ values), initial=0.0))


def _geometry_residuals(model: ForceConstants) -> tuple[float, float]:
    """Measure site-symmetry mismatch and Cartesian nonorthogonality.

    Evaluate the supplied geometry without idealizing positions or cell;
    the symmetry-search tolerance is not itself a measured error.
    """
    primitive = model.cluster_space
    symmetry = model.cluster_space.symmetry
    positions = primitive.scaled_positions
    cell = primitive.cell
    site_residual = 0.0
    orthogonality_residual = 0.0
    for operation in range(symmetry.size):
        differences = (
            positions @ symmetry.rotations[operation].T
            + symmetry.translations[operation]
            - positions[symmetry.site_permutations[operation]]
            - symmetry.site_shifts[operation]
        )
        site_residual = max(
            site_residual,
            float(np.max(np.linalg.norm(differences @ cell, axis=1), initial=0.0)),
        )
        rotation = symmetry.cartesian_rotations[operation]
        orthogonality_residual = max(
            orthogonality_residual,
            float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=2)),
        )
    return site_residual, orthogonality_residual


def enforce_rotation(
    model: ForceConstants,
    *,
    born_huang: bool = True,
    huang: bool = False,
    rank_rtol: float | None = None,
) -> RotationResult:
    """Correct FC2 rotational moments while preserving its acoustic residual.

    Born-Huang first-moment conditions describe rotational invariance when
    atoms are in force equilibrium. Huang second-moment conditions additionally
    impose zero stress and should only be enabled for that physical setting.
    The correction minimizes the Euclidean change of physical FC2 parameters
    within the resolved constraint directions. Higher orders are retained.

    Parameters
    ----------
    model
        Primitive force constants containing FC2 and their reference geometry.
        Their cluster space must have been initialized with ``asr=True``.
    born_huang
        Enforce Born-Huang first-moment conditions; enabled by default.
    huang
        Enforce zero-stress Huang second-moment conditions; disabled by default.
    rank_rtol
        Relative singular-value cutoff between zero and one. By default,
        estimate it as twice the sum of the measured position mismatch
        divided by the median pair distance and Cartesian nonorthogonality.
        A machine-precision floor is applied in either case. Directions below
        the cutoff are unresolved and are not corrected.

    Returns
    -------
    RotationResult
        New force constants, physical residuals, correction size and rank
        diagnostics for the length-normalized moment equations.

    Raises
    ------
    ValueError
        If no condition is selected, the cutoff is invalid, FC2 is absent,
        ASR coordinates were not prepared, or non-onsite pairs do not define
        a positive finite length scale.

    Notes
    -----
    The correction lies in the initialized acoustic subspace, preserving any
    existing ASR residual without measuring or repairing it. Actual
    interatomic separations enter the equations, so the projection remains
    floating-point and does not rationalize or idealize the geometry.

    References
    ----------
    C. Lin, S. Ponce and N. Marzari, npj Computational Materials 8, 236
    (2022), equations (6) and (16), doi:10.1038/s41524-022-00920-6.
    """
    if not born_huang and not huang:
        raise ValueError("select born_huang=True and/or huang=True")
    if rank_rtol is not None and (not np.isfinite(rank_rtol) or not 0 <= rank_rtol <= 1):
        raise ValueError("rank_rtol must be finite and between zero and one")
    if 2 not in model.coefficients:
        raise ValueError("rotational invariance requires FC2")
    started = perf_counter()
    logger.info(
        "Rotation projection started: born_huang=%s huang=%s rank_rtol=%s",
        born_huang,
        huang,
        rank_rtol if rank_rtol is not None else "automatic",
    )
    coordinates = model.cluster_space.acoustic_coordinates(2)
    if coordinates is None:
        raise ValueError("rotational correction requires ClusterSpace(..., asr=True)")
    born, second_moment, length_scale = _fc2_moment_matrices(model.cluster_space)
    selected = []
    if born_huang:
        selected.append(born)
    if huang:
        selected.append(second_moment)
    constraints = sparse.vstack(selected, format="csr")
    initial = model.coefficients[2]

    # Project moment rows onto the prepared subspace in the physical metric.
    # Free lattice coordinates are not an orthonormal parameterization.
    normals = coordinates.normal_basis()
    rotation_array = constraints.toarray()
    effective = rotation_array - (rotation_array @ normals) @ normals.T
    left, singular, right = np.linalg.svd(effective, full_matrices=False)
    largest = float(singular[0] if singular.size else 0.0)
    geometry_residual, orthogonality_residual = _geometry_residuals(model)
    geometry_rtol = 2.0 * (geometry_residual / length_scale + orthogonality_residual)
    relative_cutoff = geometry_rtol if rank_rtol is None else float(rank_rtol)
    cutoff = largest * max(
        relative_cutoff,
        np.finfo(float).eps * max(effective.shape),
    )
    retained = singular > cutoff
    rank = int(np.count_nonzero(retained))
    if rank:
        correction = right[retained].T @ (
            (left[:, retained].T @ (rotation_array @ initial)) / singular[retained]
        )
    else:
        correction = np.zeros_like(initial)
    projected = initial - correction
    _, relative_before = _relative_residual(constraints, initial)
    _, relative_after = _relative_residual(constraints, projected)
    correction_norm = float(np.linalg.norm(correction))
    initial_norm = float(np.linalg.norm(initial))

    coefficients = dict(model.coefficients)
    coefficients[2] = projected
    result = ForceConstants(model.cluster_space, coefficients)
    report = RotationResult(
        force_constants=result,
        born_huang=born_huang,
        huang=huang,
        length_scale=length_scale,
        equations=constraints.shape[0],
        born_huang_before=(_maximum_residual(born, initial) * length_scale if born_huang else None),
        born_huang_after=(
            _maximum_residual(born, projected) * length_scale if born_huang else None
        ),
        huang_before=(
            _maximum_residual(second_moment, initial) * length_scale**2 if huang else None
        ),
        huang_after=(
            _maximum_residual(second_moment, projected) * length_scale**2 if huang else None
        ),
        relative_before=relative_before,
        relative_after=relative_after,
        correction_norm=correction_norm,
        relative_correction=correction_norm / initial_norm if initial_norm else 0.0,
        retained_rank=rank,
        rank_rtol=relative_cutoff,
        automatic_rank=rank_rtol is None,
        rank_cutoff=cutoff,
        smallest_retained_singular_value=(float(singular[rank - 1]) if rank else None),
        largest_discarded_singular_value=(float(singular[rank]) if rank < len(singular) else None),
        geometry_residual=geometry_residual,
        orthogonality_residual=orthogonality_residual,
    )
    logger.info(
        "Rotation projection complete: retained_rank=%d rank_cutoff=%.6g "
        "relative_before=%.6g relative_after=%.6g correction_norm=%.6g "
        "elapsed_s=%.2f",
        report.retained_rank,
        report.rank_cutoff,
        report.relative_before,
        report.relative_after,
        report.correction_norm,
        perf_counter() - started,
    )
    return report


__all__ = ["RotationResult", "enforce_rotation"]
