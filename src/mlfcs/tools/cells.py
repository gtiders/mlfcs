"""Cell utilities: phonopy-replica supercell construction and spglib standardization.

The supercell builder is a dependency-free replication of phonopy's historical
algorithm; the standardization helpers hand the same structure to spglib and
report the basis change they applied.
"""

from __future__ import annotations

import operator

import numpy as np
import spglib
from ase import Atoms
from ase.data import chemical_symbols


def _matrix(values: object) -> np.ndarray:
    """Normalize supercell dimensions or a 3 by 3 matrix."""
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
    """Compute a 3x3 determinant with Python-integer intermediates, without fixed-width
    cancellation.
    """
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
    """Build an ASE supercell in primitive-site-major order.

    This function is a dependency-free replication of phonopy's historical
    supercell construction (``is_old_style=True``), so that structure
    preparation does not require phonopy at runtime.

    Algorithm
    ---------
    1. Bound the three column vectors of ``matrix.T`` with an axis-aligned
       integer box: the eight corner combinations give per-axis
       multiplicities ``m``, and ``simple_cell = diag(m) @ primitive.cell``
       is a plain supercell large enough to contain the target.
    2. Build the rational frame ``trim = matrix.T / m`` mapping the simple
       box onto the requested supercell,
       ``target_cell = trim.T @ simple_cell``.
    3. Enumerate the simple-box lattice points in phonopy's order, tile the
       wrapped fractional positions of every primitive site over them, and
       convert to target-cell fractional coordinates, wrapping by
       ``-floor`` into ``[0, 1)``.
    4. Walk the candidates in order and drop any image whose minimum-image
       distance to an earlier same-species image is below ``symprec``. The
       survivors, in site-major order (all images of site 0, then site 1,
       ...), are exactly phonopy's historical atom order.

    Parameters
    ----------
    primitive : ase.Atoms
        Fully periodic primitive structure; its wrapped fractional positions
        are the tiling seeds and its masses are carried over per site.
    matrix : sequence of int
        Three integer repetitions along the primitive axes, or a 3 by 3
        integer supercell matrix in row convention
        (``target_cell = matrix @ primitive.cell``). Entries must be
        integers and the determinant must be positive.
    symprec : float, keyword-only, default 1e-5
        Cartesian length in angstrom used to identify duplicate images where
        a lattice point lands on the boundary of the surrounding box.

    Returns
    -------
    ase.Atoms
        The supercell with ``abs(det(matrix))`` times the input atoms, in
        site-major order.

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


def standard_primitive(atoms: Atoms, *, symprec: float = 1e-5) -> Atoms:
    """Return spglib's standardized primitive cell as a new ASE Atoms object.

    The basis-change matrix from the input cell to the standardized primitive
    cell is printed in row convention (``new_cell = change @ input_cell``)
    together with its determinant, the volume ratio. Masses are carried over
    from the input, averaged per chemical element. ``no_idealize`` is left
    off, so the returned cell also sits in spglib's standard orientation.
    """
    return _standardized(atoms, to_primitive=True, symprec=symprec)


def standard_cell(atoms: Atoms, *, symprec: float = 1e-5) -> Atoms:
    """Return spglib's standardized conventional cell as a new ASE Atoms object.

    The basis-change matrix from the input cell to the standardized
    conventional cell is printed in row convention
    (``new_cell = change @ input_cell``) together with its determinant, the
    volume ratio. Masses are carried over from the input, averaged per
    chemical element. ``no_idealize`` is left off, so the returned cell also
    sits in spglib's standard orientation.
    """
    return _standardized(atoms, to_primitive=False, symprec=symprec)


def _standardized(atoms: Atoms, *, to_primitive: bool, symprec: float) -> Atoms:
    """Standardize with spglib, print the basis change, and return ASE Atoms."""
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
    label = "standard primitive cell" if to_primitive else "standard conventional cell"
    print(
        f"{label}: transformation matrix in row convention "
        f"(new_cell = change @ input_cell), det = {float(np.linalg.det(change)):.6f}"
    )
    for row in change:
        print("  [{:12.6f} {:12.6f} {:12.6f}]".format(*row))

    symbols = [chemical_symbols[number] for number in new_numbers]
    averaged: dict[str, list[float]] = {}
    for symbol, mass in zip(atoms.get_chemical_symbols(), atoms.get_masses(), strict=True):
        averaged.setdefault(symbol, []).append(mass)
    means = {symbol: float(np.mean(masses)) for symbol, masses in averaged.items()}
    return Atoms(
        numbers=new_numbers,
        masses=[means[symbol] for symbol in symbols],
        scaled_positions=new_positions,
        cell=new_cell,
        pbc=True,
    )


__all__ = ["build_supercell", "standard_cell", "standard_primitive"]
