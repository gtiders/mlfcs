"""Cell construction and crystallographic standardization utilities.

Supercells follow phonopy's historical old-style atom order without requiring
phonopy at runtime. Standard primitive and conventional cells come from
spglib, together with the basis transformation applied to the input cell.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms
from ase.data import chemical_symbols


def _matrix(values: object) -> np.ndarray:
    """Normalize diagonal repetitions or a 3x3 integer supercell matrix."""
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
    """Return the exact 3x3 determinant using Python-integer arithmetic."""
    a, b, c = (int(value) for value in matrix[0])
    d, e, f = (int(value) for value in matrix[1])
    g, h, i = (int(value) for value in matrix[2])
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _build_supercell(atoms: Atoms, matrix: np.ndarray, *, symprec: float) -> Atoms:
    """Construct a supercell in phonopy's historical atom order."""
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
    """Build a periodic ASE supercell in phonopy's historical atom order.

    ``matrix`` may be three positive integer repetitions or a nonsingular
    integer matrix in row convention,

        ``cell_super = matrix @ primitive.cell``.

    The result uses the site-major ordering of phonopy's historical
    ``is_old_style=True`` construction. Coincident boundary images are merged
    within ``symprec``, a Cartesian tolerance in angstrom. The primitive must
    be periodic in all three directions and the matrix must have positive
    determinant. The implementation uses ASE and NumPy without importing
    phonopy.
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


@dataclass(frozen=True, slots=True)
class StandardizedCell:
    """A spglib-standardized ASE cell and its basis transformation.

    ``transformation`` follows the row-vector convention
    ``atoms.cell = transformation @ input_cell``. ``atoms`` contains the
    idealized standard geometry. Masses are assigned by chemical species, so
    different isotope masses for sites of the same element are averaged.
    """

    atoms: Atoms
    transformation: np.ndarray

    def __post_init__(self) -> None:
        """Store the basis transformation as an independent readonly array."""
        transformation = np.array(self.transformation, dtype=np.float64, copy=True)
        transformation.setflags(write=False)
        object.__setattr__(self, "transformation", transformation)


def standard_primitive(atoms: Atoms, *, symprec: float = 1e-5) -> StandardizedCell:
    """Return the spglib-standardized primitive cell and its basis change.

    ``symprec`` is the Cartesian symmetry tolerance in angstrom. Spglib
    standardizes and idealizes the geometry; masses are averaged by chemical
    species. The transformation follows row convention, with
    ``result.atoms.cell = result.transformation @ atoms.cell``.
    """
    return _standardized(atoms, to_primitive=True, symprec=symprec)


def standard_conventional(atoms: Atoms, *, symprec: float = 1e-5) -> StandardizedCell:
    """Return the spglib-standardized conventional cell and its basis change.

    ``symprec`` is the Cartesian symmetry tolerance in angstrom. Spglib
    standardizes and idealizes the geometry; masses are averaged by chemical
    species. The transformation follows row convention, with
    ``result.atoms.cell = result.transformation @ atoms.cell``.
    """
    return _standardized(atoms, to_primitive=False, symprec=symprec)


def _standardized(
    atoms: Atoms, *, to_primitive: bool, symprec: float
) -> StandardizedCell:
    """Standardize a periodic structure and retain its basis transformation."""
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ASE Atoms object")
    if not np.all(atoms.pbc):
        raise ValueError("structure must be periodic in all three directions")
    symprec = float(symprec)
    if not np.isfinite(symprec) or symprec <= 0.0:
        raise ValueError(f"symprec must be a finite positive length in angstrom, got {symprec!r}")
    cell = np.asarray(atoms.cell, dtype=np.float64)
    positions = atoms.get_scaled_positions(wrap=True)
    numbers = np.asarray(atoms.numbers, dtype=np.int32)
    standardized = spglib.standardize_cell(
        (cell, positions, numbers),
        to_primitive=to_primitive,
        no_idealize=False,
        symprec=symprec,
    )
    if standardized is None:
        raise ValueError(f"spglib could not standardize the structure at symprec {symprec:g}")
    new_cell, new_positions, new_numbers = standardized
    change = np.asarray(new_cell) @ np.linalg.inv(cell)
    symbols = [chemical_symbols[number] for number in new_numbers]
    averaged: dict[str, list[float]] = {}
    for symbol, mass in zip(atoms.get_chemical_symbols(), atoms.get_masses(), strict=True):
        averaged.setdefault(symbol, []).append(mass)
    means = {symbol: float(np.mean(masses)) for symbol, masses in averaged.items()}
    return StandardizedCell(
        atoms=Atoms(
            numbers=new_numbers,
            masses=[means[symbol] for symbol in symbols],
            scaled_positions=new_positions,
            cell=new_cell,
            pbc=True,
        ),
        transformation=change,
    )


__all__ = [
    "StandardizedCell",
    "build_supercell",
    "standard_conventional",
    "standard_primitive",
]
