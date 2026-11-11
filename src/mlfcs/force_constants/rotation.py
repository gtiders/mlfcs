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

from mlfcs.cluster_space import ClusterSpace
from mlfcs.force_constants.model import ForceConstants
from mlfcs.foundation.tensors import rotate_basis
from mlfcs.geometry.idealize import IdealizedCell, idealize_reference_cell

# A structural-rank decision is accepted only when the idealized constraint
# spectrum separates the physical directions from numerical zero by at least
# this ratio. Below it the idealization has not resolved the rank; reduce
# symprec or correct the reference geometry instead of trusting the gap.
_MIN_STRUCTURAL_GAP = 1.0e3


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
    is the structural rank determined from the idealized moment spectrum.

    The idealization diagnostics record the in-basis projection residuals and
    the singular-value gap and numerical-zero bound that determine the rank.
    The reference cell keeps its basis, origin, atom count and atom order.
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
    idealization_position_residual: float
    idealization_metric_residual: float
    idealization_lattice_residual: float
    idealization_symmetry_closure: float
    structural_rank_gap: float
    structural_rank_bound: float


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
    cell atoms. Columns follow the physical FC2 parameters.
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
        for image in range(len(orbit.labels)):
            row = orbit.labels[image]
            first, second = row[0], row[1]
            vector = (
                positions[int(second[0])]
                - positions[int(first[0])]
                + np.asarray(second[1:], dtype=np.float64) @ cell
            )
            if not np.array_equal(second, first):
                lengths.append(float(np.linalg.norm(vector)))
            rotation = space.symmetry.cartesian_rotations[orbit.operations[image]].T
            basis = rotate_basis(orbit.component_basis, rotation, orbit.permutations[image])
            images.append((int(first[0]), vector, int(offsets[local_orbit]), basis))
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


def _idealized_fc2_space(space, idealized: IdealizedCell) -> ClusterSpace:
    """Build an FC2-only reference space on the idealized cell and check its layout.

    The idealized cell preserves the input basis, origin and atom order, so the
    FC2 orbit labels and parameter columns must match the original block. A
    mismatch means the idealization changed the detected symmetry or its
    canonical parameterization; the correction is then undefined.
    """
    block = space.block(2)
    atoms = space.cell_atoms.copy()
    atoms.set_cell(idealized.cell, scale_atoms=False)
    atoms.set_scaled_positions(idealized.scaled_positions)
    ideal = ClusterSpace(
        atoms,
        cutoffs={2: block.cutoff},
        max_body_orders={2: block.max_body_order},
        symprec=space.symprec,
        asr=True,
    )
    ideal_block = ideal.block(2)
    if (ideal_block.parameters.start, ideal_block.parameters.stop) != (
        block.parameters.start,
        block.parameters.stop,
    ):
        raise ValueError("idealized geometry changed the FC2 parameter layout")
    original_orbits = space.orbits[block.orbits.start : block.orbits.stop]
    ideal_orbits = ideal.orbits[ideal_block.orbits.start : ideal_block.orbits.stop]
    if len(original_orbits) != len(ideal_orbits):
        raise ValueError("idealized geometry changed the FC2 orbit count")
    for original, candidate in zip(original_orbits, ideal_orbits, strict=True):
        if original.dimension != candidate.dimension or not np.array_equal(
            original.labels, candidate.labels
        ):
            raise ValueError("idealized geometry changed the FC2 orbit layout")
    return ideal


def enforce_rotation(
    model: ForceConstants,
    *,
    born_huang: bool = True,
    huang: bool = False,
) -> RotationResult:
    """Correct FC2 rotational moments while preserving its acoustic residual.

    Born-Huang first-moment conditions describe rotational invariance when
    atoms are in force equilibrium. Huang second-moment conditions additionally
    impose zero stress and should only be enabled for that physical setting.
    The correction minimizes the Euclidean change of physical FC2 parameters
    within the resolved constraint directions. Higher orders are retained.

    The constraint rank is structural: the reference cell is first projected
    onto exact symmetry in its own basis and atom order, and the rank is read
    from the well-separated singular-value gap of that idealized moment matrix.
    The correction values still come from the measured model's moment matrix;
    no geometry-error singular-value cutoff is applied.

    Parameters
    ----------
    model
        Reference-cell force constants containing FC2 and their reference geometry.
        Their cluster space must have been initialized with ``asr=True``.
    born_huang
        Enforce Born-Huang first-moment conditions; enabled by default.
    huang
        Enforce zero-stress Huang second-moment conditions; disabled by default.

    Returns
    -------
    RotationResult
        New force constants, physical residuals, correction size, structural
        rank and in-basis idealization diagnostics.

    Raises
    ------
    ValueError
        If no condition is selected, FC2 is absent, ASR coordinates were not
        prepared, non-onsite pairs do not define a positive finite length
        scale, the idealized geometry changes the FC2 parameter layout, or the
        idealized spectrum does not separate a structural rank.

    Notes
    -----
    The correction lies in the initialized acoustic subspace, preserving any
    existing ASR residual without measuring or repairing it. The reference cell
    must be consistent with its detected space group within
    ``ClusterSpace.symprec``; the idealization residuals and structural gap are
    reported for audit.

    References
    ----------
    C. Lin, S. Ponce and N. Marzari, npj Computational Materials 8, 236
    (2022), equations (6) and (16), doi:10.1038/s41524-022-00920-6.
    """
    if not born_huang and not huang:
        raise ValueError("select born_huang=True and/or huang=True")
    if 2 not in model.coefficients:
        raise ValueError("rotational invariance requires FC2")
    started = perf_counter()
    print(f"Rotation projection: born_huang {born_huang}, huang {huang}")
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
    idealization = idealize_reference_cell(
        model.cluster_space.cell,
        model.cluster_space.scaled_positions,
        model.cluster_space.atomic_numbers,
        symprec=model.cluster_space.symprec,
    )
    ideal_space = _idealized_fc2_space(model.cluster_space, idealization)
    ideal_born, ideal_second, _ = _fc2_moment_matrices(ideal_space)
    ideal_selected = []
    if born_huang:
        ideal_selected.append(ideal_born)
    if huang:
        ideal_selected.append(ideal_second)
    ideal_array = sparse.vstack(ideal_selected, format="csr").toarray()
    ideal_normals = ideal_space.acoustic_coordinates(2).normal_basis()
    ideal_effective = ideal_array - (ideal_array @ ideal_normals) @ ideal_normals.T
    ideal_singular = np.linalg.svd(ideal_effective, compute_uv=False)
    largest_ideal = float(ideal_singular[0] if ideal_singular.size else 0.0)
    structural_bound = float(np.finfo(float).eps * max(ideal_effective.shape) * largest_ideal)
    rank = int(np.count_nonzero(ideal_singular > structural_bound))
    structural_gap = (
        float(ideal_singular[rank - 1] / ideal_singular[rank])
        if 0 < rank < len(ideal_singular) and ideal_singular[rank] > 0.0
        else float("inf")
    )
    if structural_gap < _MIN_STRUCTURAL_GAP:
        raise ValueError(
            "idealized moment equations do not separate a structural rank: "
            f"gap {structural_gap:.3g} < {_MIN_STRUCTURAL_GAP:g}; "
            "reduce symprec or correct the reference geometry"
        )
    # Rank-only: take the structural rank from the idealized geometry, but
    # keep the measured model's moment values in the correction.
    effective = rotation_array - (rotation_array @ normals) @ normals.T
    left, singular, right = np.linalg.svd(effective, full_matrices=False)
    largest = float(singular[0] if singular.size else 0.0)
    real_bound = float(np.finfo(float).eps * max(effective.shape) * largest)
    if rank and singular[rank - 1] <= real_bound:
        raise ValueError("real moment matrix does not resolve the idealized structural rank")
    retained = np.zeros(len(singular), dtype=bool)
    retained[:rank] = True
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
        idealization_position_residual=idealization.position_residual,
        idealization_metric_residual=idealization.metric_residual,
        idealization_lattice_residual=idealization.lattice_residual,
        idealization_symmetry_closure=idealization.symmetry_closure,
        structural_rank_gap=structural_gap,
        structural_rank_bound=structural_bound,
    )
    print(
        f"Rotation projection complete: retained rank {report.retained_rank}, "
        f"structural gap {report.structural_rank_gap:.3g}, "
        f"relative {report.relative_before:.3g} -> {report.relative_after:.3g}, "
        f"correction {report.correction_norm:.3g}, {perf_counter() - started:.2f} s"
    )
    return report


__all__ = ["RotationResult", "enforce_rotation"]
