"""Build ASE supercells in phonopy's old-style atom ordering."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from mlfcs.structure.integer_lattice import determinant_3x3, normalize_supercell_matrix


def _phonopy_atoms(atoms: Atoms):
    from phonopy.structure.atoms import PhonopyAtoms

    kwargs = {
        "symbols": atoms.get_chemical_symbols(),
        "cell": np.asarray(atoms.cell),
        "scaled_positions": atoms.get_scaled_positions(wrap=True),
    }
    masses = atoms.get_masses()
    if masses is not None:
        kwargs["masses"] = masses
    return PhonopyAtoms(**kwargs)


def _from_phonopy(atoms: Atoms, matrix: np.ndarray, *, symprec: float) -> Atoms:
    from phonopy.structure.cells import get_supercell

    result = get_supercell(_phonopy_atoms(atoms), matrix.T, is_old_style=True, symprec=symprec)
    return Atoms(
        symbols=result.symbols,
        cell=np.asarray(result.cell),
        scaled_positions=np.asarray(result.scaled_positions),
        pbc=True,
    )


def _fallback_phonopy_old_style(atoms: Atoms, matrix: np.ndarray, *, symprec: float) -> Atoms:
    determinant = determinant_3x3(matrix)
    if determinant <= 0:
        raise ValueError("phonopy ordering requires a positive determinant")
    phonopy_matrix = matrix.T
    corners = np.asarray(
        (
            (0, 0, 0),
            phonopy_matrix[:, 0],
            phonopy_matrix[:, 1],
            phonopy_matrix[:, 2],
            phonopy_matrix[:, 1] + phonopy_matrix[:, 2],
            phonopy_matrix[:, 2] + phonopy_matrix[:, 0],
            phonopy_matrix[:, 0] + phonopy_matrix[:, 1],
            phonopy_matrix[:, 0] + phonopy_matrix[:, 1] + phonopy_matrix[:, 2],
        ),
        dtype=np.int64,
    )
    multiplicities = np.max(corners, axis=0) - np.min(corners, axis=0)
    if np.any(multiplicities <= 0):
        raise ValueError("phonopy surrounding frame has a zero multiplicity")
    simple_matrix = np.diag(multiplicities)
    simple_cell = simple_matrix @ np.asarray(atoms.cell)
    trim_frame = phonopy_matrix / multiplicities[:, None]
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
    """Build an ASE reference supercell in phonopy's old-style ordering.

    This is a structure-generation utility only.  Calculation APIs never
    invoke it implicitly: pass its result as the explicit reference
    supercell when that ordering is required.
    """
    if not isinstance(primitive, Atoms):
        raise TypeError("primitive must be an ASE Atoms object")
    if not np.all(primitive.pbc):
        raise ValueError("primitive must be periodic")
    matrix = normalize_supercell_matrix(supercell_matrix)
    try:
        return _from_phonopy(primitive, matrix, symprec=symprec)
    except ImportError:
        return _fallback_phonopy_old_style(primitive, matrix, symprec=symprec)


__all__ = ["build_supercell"]
