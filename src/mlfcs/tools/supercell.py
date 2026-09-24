"""Optional ASE structure-generation helpers."""

from __future__ import annotations

import operator

import numpy as np
from ase import Atoms


def _matrix(values: object) -> np.ndarray:
    """Normalize an exact integer triple or 3 by 3 matrix."""
    array = np.asarray(values, dtype=object)
    if array.shape == (3,):
        array = np.diag(array)
    if array.shape != (3, 3):
        raise ValueError(f"supercell matrix must have shape (3,) or (3, 3), got {array.shape}")
    rows = []
    for row_index, row in enumerate(array.tolist()):
        converted = []
        for column_index, value in enumerate(row):
            try:
                converted.append(operator.index(value))
            except TypeError as error:
                raise TypeError(
                    "supercell matrix entries must be declared as integers; "
                    f"entry ({row_index}, {column_index}) is {value!r}"
                ) from error
        rows.append(converted)
    try:
        return np.asarray(rows, dtype=np.int64)
    except OverflowError as error:
        raise OverflowError("supercell matrix entries must fit ASE's int64 matrix") from error


def _determinant(matrix: np.ndarray) -> int:
    a, b, c = (int(value) for value in matrix[0])
    d, e, f = (int(value) for value in matrix[1])
    g, h, i = (int(value) for value in matrix[2])
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _build_supercell(atoms: Atoms, matrix: np.ndarray, *, symprec: float) -> Atoms:
    """Enumerate a surrounding integer box in phonopy's old-style order."""
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
        np.tile(lattice_points, (len(atoms), 1))
        + np.repeat(scaled, images, axis=0)
    ) @ np.linalg.inv(simple_matrix).T
    positions = positions @ np.linalg.inv(trim_frame).T
    positions -= np.floor(positions)
    numbers = np.repeat(atoms.numbers, images)
    masses = np.repeat(atoms.get_masses(), images)

    selected: list[int] = []
    for atom, position in enumerate(positions):
        if selected:
            previous = np.asarray(selected)
            delta = positions[previous] - position
            delta -= np.rint(delta)
            distances = np.linalg.norm(delta @ target_cell, axis=1)
            same_species = numbers[previous] == numbers[atom]
            if np.any((distances < symprec) & same_species):
                continue
        selected.append(atom)

    return Atoms(
        numbers=numbers[selected],
        masses=masses[selected],
        scaled_positions=positions[selected],
        cell=target_cell,
        pbc=True,
    )


def build_supercell(
    primitive: Atoms,
    matrix: object,
    *,
    symprec: float = 1e-5,
) -> Atoms:
    """Build an ASE supercell in primitive-site-major order.

    This is an optional input-preparation helper. Core calculation objects take
    an explicit ``Atoms`` supercell and never call it implicitly. ``matrix``
    may be three integer repetitions or a 3 by 3 integer supercell matrix. The
    returned atom order follows phonopy's historical ``is_old_style=True``
    order, including the lattice-image enumeration for general matrices.
    ``symprec`` is the Cartesian length used to identify duplicate images on
    the boundary of the temporary surrounding box.

    The implementation uses NumPy and ASE only; it has no phonopy dependency.
    """
    if not isinstance(primitive, Atoms):
        raise TypeError("primitive must be an ASE Atoms object")
    if not np.all(primitive.pbc):
        raise ValueError("primitive must be periodic in all three directions")
    supercell_matrix = _matrix(matrix)
    determinant = _determinant(supercell_matrix)
    if determinant <= 0:
        raise ValueError("supercell construction requires a positive determinant")
    symprec = float(symprec)
    if not np.isfinite(symprec) or symprec <= 0.0:
        raise ValueError(f"symprec must be a finite positive length in angstrom, got {symprec!r}")
    return _build_supercell(primitive, supercell_matrix, symprec=symprec)


__all__ = ["build_supercell"]
