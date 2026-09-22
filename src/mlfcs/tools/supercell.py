"""Build and align explicit ASE supercells outside the calculation core."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.optimize import linear_sum_assignment

from mlfcs.structure.integer_lattice import determinant_3x3, normalize_supercell_matrix
from mlfcs.structure.periodic_geometry import PeriodicGeometry


def _is_integer_matrix(matrix: object) -> bool:
    """Return whether every entry of a candidate matrix is a Python or NumPy integer."""
    values = np.asarray(matrix)
    if values.dtype == object:
        return all(isinstance(value, (int, np.integer)) for value in values.ravel())
    return bool(np.issubdtype(values.dtype, np.integer))


def _build_supercell(atoms: Atoms, matrix: np.ndarray, *, symprec: float) -> Atoms:
    """Construct a primitive-site-major supercell using only NumPy and ASE."""
    determinant = determinant_3x3(matrix)
    if determinant <= 0:
        raise ValueError("supercell construction requires a positive determinant")
    column_matrix = matrix.T
    corners = np.asarray(
        (
            (0, 0, 0),
            column_matrix[:, 0],
            column_matrix[:, 1],
            column_matrix[:, 2],
            column_matrix[:, 1] + column_matrix[:, 2],
            column_matrix[:, 2] + column_matrix[:, 0],
            column_matrix[:, 0] + column_matrix[:, 1],
            column_matrix[:, 0] + column_matrix[:, 1] + column_matrix[:, 2],
        ),
        dtype=np.int64,
    )
    multiplicities = np.max(corners, axis=0) - np.min(corners, axis=0)
    if np.any(multiplicities <= 0):
        raise ValueError("supercell surrounding frame has a zero multiplicity")
    simple_matrix = np.diag(multiplicities)
    simple_cell = simple_matrix @ np.asarray(atoms.cell)
    trim_frame = column_matrix / multiplicities[:, None]
    target_cell = trim_frame.T @ simple_cell
    b, c, a = np.meshgrid(
        range(int(multiplicities[1])),
        range(int(multiplicities[2])),
        range(int(multiplicities[0])),
    )
    lattice_points = np.c_[a.ravel(), b.ravel(), c.ravel()]
    images = len(lattice_points)
    scaled = atoms.get_scaled_positions(wrap=True)
    positions = (
        np.tile(lattice_points, (len(atoms), 1)) + np.repeat(scaled, images, axis=0)
    ) @ np.linalg.inv(simple_matrix).T
    positions = positions @ np.linalg.inv(trim_frame).T
    positions -= np.floor(positions)
    numbers = np.repeat(atoms.numbers, images)
    masses = np.repeat(atoms.get_masses(), images)
    selected: list[int] = []
    for atom, position in enumerate(positions):
        if selected:
            delta = positions[np.asarray(selected)] - position
            delta -= np.rint(delta)
            distance = np.linalg.norm(delta @ target_cell, axis=1)
            if np.any((distance < symprec) & (numbers[np.asarray(selected)] == numbers[atom])):
                continue
        selected.append(atom)
    return Atoms(
        numbers=numbers[np.asarray(selected)],
        masses=masses[np.asarray(selected)],
        scaled_positions=positions[np.asarray(selected)],
        cell=target_cell,
        pbc=True,
    )


def build_supercell(
    primitive: Atoms,
    supercell_matrix: object,
    *,
    symprec: float = 1e-5,
) -> Atoms:
    """Build an ASE reference supercell in primitive-site-major ordering.

    This is an optional structure-generation utility only. Calculation APIs never invoke it
    implicitly: the caller may build a supercell here, read one from a file, or obtain one from
    another program, then passes that explicit structure as ``reference`` in the next step.

    ``supercell_matrix`` is a discrete construction parameter, so a floating-point form such as
    ``[[2.0, 0.0, 0.0], ...]`` is refused: that is not a numerical approximation question but a
    statement about what the caller meant. ``symprec`` is the Cartesian length used to
    de-duplicate generated sites on the boundary of the surrounding frame.
    """
    if not isinstance(primitive, Atoms):
        raise TypeError("primitive must be an ASE Atoms object")
    if not np.all(primitive.pbc):
        raise ValueError("primitive must be periodic")
    symprec = float(symprec)
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError(f"symprec must be a finite positive length in angstrom, got {symprec!r}")
    if not _is_integer_matrix(supercell_matrix):
        raise TypeError(
            "supercell_matrix must be an integer matrix, integer triple or a nested sequence of "
            f"Python/NumPy integers; got {supercell_matrix!r}"
        )
    matrix = normalize_supercell_matrix(supercell_matrix)
    return _build_supercell(primitive, matrix, symprec=symprec)


def align_structures(
    reference: Atoms,
    atoms: Atoms,
    *,
    tolerance: float,
) -> tuple[Atoms, float]:
    """Reorder an external structure to ``reference`` and report its residual.

    This is an explicit external-import policy. Calculation APIs never invoke it
    implicitly and never silently reorder a training frame.
    """
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            f"tolerance must be a finite positive length in angstrom, got {tolerance!r}"
        )
    if len(atoms) != len(reference):
        raise ValueError("structure atom count differs from reference")
    cell_residual = float(
        np.max(np.linalg.norm(np.asarray(atoms.cell) - np.asarray(reference.cell), axis=1))
    )
    if cell_residual >= tolerance:
        raise ValueError(
            "structure cell differs from reference: lattice residual "
            f"{cell_residual:.6e} angstrom against tolerance {tolerance:.6e} angstrom"
        )
    permutation = np.empty(len(reference), dtype=np.int32)
    maximum = 0.0
    geometry = PeriodicGeometry(reference.cell, reference.pbc)
    for number in np.unique(reference.numbers):
        target = np.flatnonzero(reference.numbers == number)
        source = np.flatnonzero(atoms.numbers == number)
        if len(target) != len(source):
            raise ValueError("structure chemical composition differs from reference")
        delta = atoms.positions[source][None, :, :] - reference.positions[target][:, None, :]
        _, lengths = geometry.mic(delta.reshape(-1, 3))
        cost = lengths.reshape(len(target), len(source))
        rows, columns = linear_sum_assignment(cost)
        maximum = max(maximum, float(np.max(cost[rows, columns], initial=0.0)))
        permutation[target[rows]] = source[columns]
    if maximum >= tolerance:
        raise ValueError(
            "structure cannot be aligned to reference within tolerance; maximum residual "
            f"{maximum:.3e} angstrom"
        )
    aligned = atoms[permutation]
    aligned.info.update(atoms.info)
    return aligned, max(cell_residual, maximum)


__all__ = ["align_structures", "build_supercell"]
