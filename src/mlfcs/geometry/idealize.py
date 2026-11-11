"""In-basis idealization of an arbitrary periodic reference cell.

The rotation conditions used by :mod:`mlfcs.force_constants.rotation` involve
real interatomic distances.  When the reference structure is known only up to
a symmetry-matching tolerance, those distances carry a small symmetry mismatch
that makes the constraint matrix numerically rank-deficient at the mismatch
level.  This module projects the supplied cell onto its exact symmetry in the
input basis, preserving the cell basis, origin, atom count and atom order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib

# A symmetry operation must map each atom to an atom of the same species.  The
# nearest-image mismatch is allowed to be this many times ``symprec`` before
# the supplied structure is rejected as inconsistent with the detected group.
_MATCH_FACTOR = 10.0


@dataclass(frozen=True, slots=True)
class IdealizedCell:
    """Exact-symmetry cell obtained by in-basis projection.

    ``cell`` and ``scaled_positions`` use the same lattice basis, origin, atom
    count and atom order as the input.  Only the metric and the atomic
    positions are adjusted; no standard-setting transformation, atom
    reordering or origin shift is applied.

    ``position_residual`` is the largest Cartesian displacement in angstrom
    between the input atoms and their idealized images.  ``metric_residual``
    and ``lattice_residual`` are relative Frobenius changes of the metric
    tensor and the lattice matrix.  ``symmetry_closure`` is the largest
    residual of the generating operations on the idealized structure.
    """

    cell: np.ndarray
    scaled_positions: np.ndarray
    atomic_numbers: np.ndarray
    spacegroup: int
    operations: int
    symprec: float
    position_residual: float
    metric_residual: float
    lattice_residual: float
    symmetry_closure: float


def _minimum_image(delta: np.ndarray) -> np.ndarray:
    """Return fractional differences reduced to the nearest periodic image."""
    return delta - np.round(delta)


def idealize_reference_cell(
    cell: np.ndarray,
    scaled_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    symprec: float = 1e-5,
) -> IdealizedCell:
    """Project a periodic reference cell onto exact symmetry in its own basis.

    Spglib supplies the space-group operations of the supplied cell.  The
    fractional metric tensor is averaged over the point-group operations and
    the lattice matrix closest to the input (same basis and orientation) is
    recovered by a symmetric square root.  Each crystallographic orbit is then
    fitted to the symmetry-constrained positions: every symmetry image of the
    representative is back-rotated and averaged, which enforces site symmetry
    for special positions and determines the free parameters of general
    positions.

    Parameters
    ----------
    cell
        Row-vector lattice matrix in angstrom.  The basis is retained.
    scaled_positions
        Fractional coordinates with shape ``(n_atoms, 3)``.  The atom order is
        retained.
    atomic_numbers
        Integer species identifiers with shape ``(n_atoms,)``.
    symprec
        Cartesian symmetry matching tolerance in angstrom.

    Returns
    -------
    IdealizedCell
        The idealized cell in the input basis and atom order, together with
        projection residuals.

    Raises
    ------
    ValueError
        If the inputs are malformed, the symmetry search fails, an operation
        cannot be matched species-wise within ``symprec``, or the projected
        metric is not positive definite.
    """
    cell = np.asarray(cell, dtype=np.float64)
    scaled_positions = np.asarray(scaled_positions, dtype=np.float64)
    atomic_numbers = np.asarray(atomic_numbers)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("cell must be a finite 3x3 matrix")
    if np.linalg.det(cell) == 0.0:
        raise ValueError("cell must be invertible")
    if scaled_positions.ndim != 2 or scaled_positions.shape[1] != 3:
        raise ValueError("scaled_positions must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(scaled_positions)):
        raise ValueError("scaled_positions must be finite")
    n_atoms = len(scaled_positions)
    if atomic_numbers.shape != (n_atoms,):
        raise ValueError("atomic_numbers must have shape (n_atoms,)")
    if not np.isfinite(symprec) or symprec <= 0.0:
        raise ValueError("symprec must be a positive finite length")

    dataset = spglib.get_symmetry_dataset(
        (cell, scaled_positions, atomic_numbers), symprec=float(symprec)
    )
    if dataset is None:
        raise ValueError("symmetry search failed for the supplied cell")
    rotations = np.asarray(dataset.rotations, dtype=np.float64)
    translations = np.asarray(dataset.translations, dtype=np.float64)
    operations = len(rotations)

    # Lattice: average the fractional metric over the point group, then choose
    # the lattice matrix closest to the input with the averaged metric.
    metric = cell @ cell.T
    metric_ideal = np.mean([rotation.T @ metric @ rotation for rotation in rotations], axis=0)
    metric_ideal = 0.5 * (metric_ideal + metric_ideal.T)
    inverse_cell = np.linalg.inv(cell)
    square = inverse_cell @ metric_ideal @ inverse_cell.T
    square = 0.5 * (square + square.T)
    eigenvalues, eigenvectors = np.linalg.eigh(square)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if np.min(eigenvalues) < -1e-12 * scale:
        raise ValueError("projected metric is not positive definite")
    root = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T
    cell_ideal = cell @ root
    cartesian = cell_ideal.T

    # Cartesian operations for the idealized cell.  The fractional convention
    # is x' = x R^T + t for row coordinates.
    cartesian_rotations = np.empty_like(rotations)
    cartesian_translations = np.empty((operations, 3))
    for operation in range(operations):
        cartesian_rotations[operation] = (
            cartesian @ rotations[operation] @ np.linalg.inv(cartesian)
        )
        cartesian_translations[operation] = cartesian @ translations[operation]
        # Metric projection leaves the linear parts orthogonal up to roundoff.
        u, _, vh = np.linalg.svd(cartesian_rotations[operation])
        cartesian_rotations[operation] = u @ vh

    def cartesian_delta(delta: np.ndarray) -> np.ndarray:
        return _minimum_image(delta) @ cartesian.T

    # Species-aware atom permutations for every operation.  The periodic
    # nearest-image match is vectorized over atoms; a raised error means the
    # supplied structure is inconsistent with the detected group.
    permutations = np.empty((operations, n_atoms), dtype=np.int64)
    atom_index = np.arange(n_atoms)
    for operation in range(operations):
        target = (scaled_positions @ rotations[operation].T + translations[operation]) % 1.0
        delta = target[:, None, :] - scaled_positions[None, :, :]
        delta -= np.round(delta)
        offsets = delta @ cell.T
        distance = np.linalg.norm(offsets, axis=2)
        same_species = atomic_numbers[:, None] == atomic_numbers[None, :]
        distance = np.where(same_species, distance, np.inf)
        best = np.argmin(distance, axis=1)
        best_distance = distance[atom_index, best]
        worst = float(np.max(best_distance))
        if worst > _MATCH_FACTOR * symprec:
            raise ValueError(
                f"operation {operation} cannot match all atoms within "
                f"{_MATCH_FACTOR * symprec:g} angstrom (worst {worst:g}): "
                "the structure is inconsistent with the detected space group"
            )
        permutations[operation] = best

    # Fit each orbit representative, then regenerate the orbit in input order.
    columns = scaled_positions.T
    columns_ideal = columns.copy()
    equivalents = np.asarray(dataset.equivalent_atoms)
    for representative in np.unique(equivalents):
        members = np.flatnonzero(equivalents == representative)
        if len(members) == 0:
            continue
        reference = int(members[0])
        predictions = []
        for operation in range(operations):
            atom = permutations[operation, reference]
            predicted = np.linalg.solve(
                rotations[operation], columns[:, atom] - translations[operation]
            )
            predicted = predicted + np.round(columns[:, reference] - predicted)
            predictions.append(cartesian @ predicted)
        average = np.mean(predictions, axis=0)
        representative_ideal = np.linalg.solve(cartesian, average)
        operations_to_member = {int(member): [] for member in members}
        for operation in range(operations):
            operations_to_member[int(permutations[operation, reference])].append(operation)
        for atom in members:
            best_distance = np.inf
            best_position = columns_ideal[:, atom]
            for operation in operations_to_member[int(atom)]:
                position = rotations[operation] @ representative_ideal + translations[operation]
                position = position + np.round(columns[:, atom] - position)
                distance = np.linalg.norm(
                    cartesian_delta(position - columns[:, atom])
                )
                if distance < best_distance:
                    best_distance = distance
                    best_position = position
            if not np.isfinite(best_distance):
                raise ValueError(
                    f"no operation maps orbit representative {reference} to atom {int(atom)}"
                )
            columns_ideal[:, atom] = best_position

    scaled_positions_ideal = (columns_ideal.T) % 1.0
    scaled_positions_ideal = np.where(
        np.isclose(scaled_positions_ideal, 1.0, atol=1e-12), 0.0, scaled_positions_ideal
    )

    # Diagnostics.
    position_residual = max(
        np.linalg.norm(
            cartesian_delta(scaled_positions_ideal[atom] - scaled_positions[atom])
        )
        for atom in range(n_atoms)
    )
    closure = 0.0
    for operation in range(operations):
        for atom in range(n_atoms):
            target = rotations[operation] @ columns_ideal[:, atom] + translations[operation]
            closure = max(
                closure,
                np.linalg.norm(
                    cartesian_delta(
                        target - columns_ideal[:, permutations[operation, atom]]
                    )
                ),
            )
    metric_residual = float(
        np.linalg.norm(metric - metric_ideal) / np.linalg.norm(metric)
    )
    lattice_residual = float(
        np.linalg.norm(cell_ideal - cell) / np.linalg.norm(cell)
    )

    return IdealizedCell(
        cell=np.array(cell_ideal, dtype=np.float64, copy=True),
        scaled_positions=np.array(scaled_positions_ideal, dtype=np.float64, copy=True),
        atomic_numbers=np.array(atomic_numbers, copy=True),
        spacegroup=int(dataset.number),
        operations=operations,
        symprec=float(symprec),
        position_residual=float(position_residual),
        metric_residual=metric_residual,
        lattice_residual=lattice_residual,
        symmetry_closure=float(closure),
    )
