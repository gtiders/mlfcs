"""Primitive atomic structures and periodic site addresses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms

from mlfcs._arrays import as_int64_array, readonly


def validate_primitive_arrays(cell, scaled_positions, atomic_numbers, symprec):
    """Validate a periodic primitive motif and return normalized readonly arrays.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
        Lattice vectors as rows, in angstrom.
    scaled_positions : array_like, shape (n_atoms, 3)
        Fractional row coordinates already wrapped into [0, 1); values outside this interval are rejected.
    atomic_numbers : array_like, shape (n_atoms,)
        Declared integer species identifiers representable by spglib's int32 ABI.
    symprec : float
        Positive Cartesian matching tolerance in angstrom.

    Returns
    -------
    cell, positions, numbers, symprec : tuple
        Readonly float64 geometry, int64 species and normalized scalar tolerance.

    Raises
    ------
    ValueError
        Geometry is invalid or spglib cannot certify the declared atom count as primitive.
    TypeError
        Species are not declared integers.
    OverflowError
        Species or array sizes cannot enter the required integer interfaces.

    Notes
    -----
    The input basis and atom order are retained; no standardized cell is substituted.
    """
    cell = readonly(cell, np.float64)
    positions = np.asarray(scaled_positions, dtype=np.float64)
    numbers = np.asarray(atomic_numbers)
    symprec = float(symprec)
    if cell.shape != (3, 3):
        raise ValueError("primitive cell must have shape (3, 3)")
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("primitive scaled positions must have shape (n, 3)")
    if numbers.shape != (len(positions),):
        raise ValueError("primitive atomic numbers must have shape (n,)")
    if len(numbers) == 0:
        raise ValueError("primitive cell must contain at least one atom")
    if not np.all(np.isfinite(cell)) or not np.all(np.isfinite(positions)):
        raise ValueError("primitive cell and positions must be finite")
    if not np.isfinite(symprec) or symprec <= 0.0:
        raise ValueError("symprec must be a positive finite length in angstrom")
    if float(np.linalg.det(cell)) == 0.0:
        raise ValueError("primitive cell must be nonsingular")
    if not np.issubdtype(numbers.dtype, np.integer):
        raise TypeError("primitive atomic numbers must be integers")
    numbers = as_int64_array(numbers, name="primitive atomic numbers")
    if int(numbers.min()) < np.iinfo(np.int32).min or int(numbers.max()) > np.iinfo(np.int32).max:
        raise OverflowError("primitive atomic numbers exceed the spglib int32 interface")
    if np.any(positions < 0.0) or np.any(positions >= 1.0):
        raise ValueError("primitive scaled positions must be wrapped into [0, 1)")
    found = spglib.find_primitive(
        (cell, positions, numbers.astype(np.int32, copy=False)),
        symprec=symprec,
    )
    if found is None:
        raise ValueError(
            f"spglib could not certify a primitive cell at symprec {symprec:g} angstrom"
        )
    if len(found[2]) != len(numbers):
        raise ValueError(
            f"input contains {len(numbers)} atoms but its primitive cell contains "
            f"{len(found[2])} atoms at symprec {symprec:g} angstrom"
        )

    return cell, readonly(positions, np.float64), numbers, symprec


def primitive_data(atoms, symprec):
    """Extract validated primitive geometry and positive masses from ASE Atoms.

    Requires full three-dimensional periodicity and uses ASE get_scaled_positions(wrap=True).
    Returns readonly geometry arrays
    and masses in atomic mass units; it does not retain or mutate ``atoms``.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError("primitive_atoms must be an ASE Atoms object")
    if not np.all(atoms.pbc):
        raise ValueError("primitive cell must be periodic in all three directions")
    cell, positions, numbers, symprec = validate_primitive_arrays(
        atoms.cell.array, atoms.get_scaled_positions(wrap=True), atoms.numbers, symprec
    )
    masses = readonly(atoms.get_masses(), np.float64)
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("atomic masses must be positive and finite")
    return {
        "cell": cell,
        "scaled_positions": positions,
        "atomic_numbers": numbers,
        "symprec": symprec,
        "masses": masses,
    }


@dataclass(frozen=True, order=True, slots=True)
class LatticeSite:
    """Identify one motif atom and one of its periodic images.

    Parameters
    ----------
    site : int
        Nonnegative zero-based index into the primitive motif. Consumers check
        that it is within the motif.
    translation : tuple of int, default (0, 0, 0)
        Three lattice-coordinate offsets along the primitive cell vectors,
        whose vectors are stored as rows. The default identifies the atom in
        the reference cell; nonzero offsets identify its periodic images.

    Notes
    -----
    The value is immutable, orderable and hashable. Translation is not a
    Cartesian displacement. Distinct images remain distinct even when they
    refer to the same motif atom.
    """

    site: int
    translation: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        """Validate the motif index and the three lattice-coordinate offsets."""
        if self.site < 0:
            raise ValueError("primitive site must be non-negative")
        if len(self.translation) != 3 or any(
            not isinstance(value, (int, np.integer)) for value in self.translation
        ):
            raise TypeError("lattice translation must contain exactly three integers")
        object.__setattr__(self, "site", int(self.site))
        object.__setattr__(self, "translation", tuple(int(value) for value in self.translation))
