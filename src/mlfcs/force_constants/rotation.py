"""Resolvable FC2 rotational moments, independent of ASR projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from mlfcs.force_constants.acoustic import constraint_matrix, relative_residual
from mlfcs.force_constants.lattice import rotate_basis
from mlfcs.force_constants.model import ForceConstants


@dataclass(frozen=True, slots=True)
class RotationResult:
    """An FC2 model projected onto resolvable rotational conditions."""

    force_constants: ForceConstants
    born_huang: bool
    huang: bool
    length_scale: float
    equations: int
    acoustic_before: float
    acoustic_after: float
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
    rows, columns, data = entries
    for column, value in enumerate(values):
        if value != 0.0:
            rows.append(row)
            columns.append(column_start + column)
            data.append(float(value))


def _fc2_moment_matrices(
    model: ForceConstants,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, float]:
    space = model.space
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
    positions = space.primitive.cartesian_positions
    cell = space.primitive.cell
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
        shape=(space.primitive.size * 9, parameter_count),
    ).tocsr()
    huang = sparse.coo_matrix(
        (huang_entries[2], (huang_entries[0], huang_entries[1])),
        shape=(81, parameter_count),
    ).tocsr()
    for matrix in (born, huang):
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
    return born, huang, length_scale


def _maximum_residual(matrix: sparse.csr_matrix, values: np.ndarray) -> float:
    if matrix.shape[0] == 0:
        return 0.0
    return float(np.max(np.abs(matrix @ values), initial=0.0))


def _geometry_residuals(model: ForceConstants) -> tuple[float, float]:
    """Measure the symmetry mismatch of the supplied, non-idealized primitive."""
    primitive = model.space.primitive
    symmetry = model.space.symmetry
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
    """Project FC2 onto resolvable rotational moments without changing its ASR.

    Huang's second-moment condition is appropriate only for a stress-free
    equilibrium structure and therefore has to be enabled explicitly. The
    automatic singular-value cutoff is estimated from *observed* geometry
    mismatch, not the caller's symmetry-search precision. ``rank_rtol``
    overrides it as a fraction of the largest singular value.
    """
    if not born_huang and not huang:
        raise ValueError("select born_huang=True and/or huang=True")
    if rank_rtol is not None and (not np.isfinite(rank_rtol) or not 0 <= rank_rtol <= 1):
        raise ValueError("rank_rtol must be finite and between zero and one")
    if 2 not in model.coefficients:
        raise ValueError("rotational invariance requires FC2")
    acoustic = constraint_matrix(model, 2)
    born, second_moment, length_scale = _fc2_moment_matrices(model)
    selected = []
    if born_huang:
        selected.append(born)
    if huang:
        selected.append(second_moment)
    constraints = sparse.vstack(selected, format="csr")
    initial = model.coefficients[2]

    # The acoustic equations are used only to restrict the *change*: this
    # operation neither applies ASR nor changes a pre-existing ASR residual.
    acoustic_array = acoustic.toarray()
    _, acoustic_singular, acoustic_right = np.linalg.svd(acoustic_array, full_matrices=False)
    acoustic_cutoff = (
        np.finfo(float).eps
        * max(acoustic_array.shape)
        * float(acoustic_singular[0] if acoustic_singular.size else 0.0)
    )
    acoustic_right = acoustic_right[acoustic_singular > acoustic_cutoff]
    rotation_array = constraints.toarray()
    effective = rotation_array - (rotation_array @ acoustic_right.T) @ acoustic_right
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
    _, relative_before = relative_residual(constraints, initial)
    _, relative_after = relative_residual(constraints, projected)
    correction_norm = float(np.linalg.norm(correction))
    initial_norm = float(np.linalg.norm(initial))

    coefficients = dict(model.coefficients)
    coefficients[2] = projected
    result = ForceConstants(model.space, coefficients)
    return RotationResult(
        force_constants=result,
        born_huang=born_huang,
        huang=huang,
        length_scale=length_scale,
        equations=constraints.shape[0],
        acoustic_before=_maximum_residual(acoustic, initial),
        acoustic_after=_maximum_residual(acoustic, projected),
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


__all__ = ["RotationResult", "enforce_rotation"]
