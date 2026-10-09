"""Reference-cell geometry and periodic lattice-site addresses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs.foundation.arrays import as_int64_array, readonly


def validate_primitive_arrays(cell, scaled_positions, atomic_numbers, symprec):
    """Validate a periodic reference structure without changing its cell basis.

    The structure is defined by row lattice vectors ``cell``, wrapped
    fractional motif coordinates, and atomic species. The supplied cell
    defines the reference lattice, whether or not a smaller primitive cell
    exists. No primitiveness check, reduction or standardization is performed.

    Parameters
    ----------
    cell
        Lattice vectors as rows, shape ``(3, 3)``, in angstrom.
    scaled_positions
        Fractional motif coordinates of shape ``(n_atoms, 3)``, wrapped into
        ``[0, 1)``.
    atomic_numbers
        Atomic numbers of shape ``(n_atoms,)``.
    symprec
        Positive Cartesian tolerance in angstrom for subsequent symmetry matching.

    Returns
    -------
    cell
        Readonly ``float64`` lattice matrix.
    scaled_positions
        Readonly ``float64`` fractional coordinates.
    atomic_numbers
        Readonly ``int64`` atomic numbers.
    symprec
        Normalized positive floating-point tolerance.

    Raises
    ------
    TypeError
        If atomic numbers are not represented by integers.
    ValueError
        If the geometry is invalid or fractional coordinates are not wrapped.
    OverflowError
        If atomic numbers cannot enter the integer interface required by
        spglib.
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
    return cell, readonly(positions, np.float64), numbers, symprec


def primitive_data(atoms, symprec):
    """Extract the periodic reference-cell representation from an ASE structure.

    The structure must be fully periodic in three dimensions. Its cell need
    not be primitive. Fractional coordinates are wrapped into the reference cell,
    while the original lattice basis and atom order are preserved. Atomic
    masses are returned in atomic mass units and must be positive and finite.
    The input ``Atoms`` object is neither retained nor modified.
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
    """Address one atomic site of the infinite periodic crystal.

    A site ``(i, n)`` denotes primitive motif atom ``i`` translated by the
    integer lattice vector ``n``. With row lattice vectors,

        r(i, n) = (s_i + n) @ cell.

    ``translation`` is therefore a lattice-coordinate address, not a
    Cartesian displacement.
    """

    site: int
    translation: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        """Validate and normalize the lattice-site address."""
        if self.site < 0:
            raise ValueError("primitive site must be non-negative")
        if len(self.translation) != 3 or any(
            not isinstance(value, (int, np.integer)) for value in self.translation
        ):
            raise TypeError("lattice translation must contain exactly three integers")
        object.__setattr__(self, "site", int(self.site))
        object.__setattr__(self, "translation", tuple(int(value) for value in self.translation))
