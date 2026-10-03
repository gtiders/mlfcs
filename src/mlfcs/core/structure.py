"""Validated atomic structures and exact primitive lattice addresses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms

from mlfcs._arrays import integer_array, readonly


def validate_primitive_arrays(cell, scaled_positions, atomic_numbers, symprec):
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
    numbers = integer_array(numbers, name="primitive atomic numbers")
    if int(numbers.min()) < np.iinfo(np.int32).min or int(numbers.max()) > np.iinfo(np.int32).max:
        raise OverflowError("primitive atomic numbers exceed the spglib int32 interface")
    wrapped = np.mod(positions, 1.0)
    wrapped[wrapped == 1.0] = 0.0
    found = spglib.find_primitive(
        (cell, wrapped, numbers.astype(np.int32, copy=False)),
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

    return cell, readonly(wrapped, np.float64), numbers, symprec


def structure_fingerprint(cell, scaled_positions, atomic_numbers, symprec) -> str:
    """Stable identity of the declared primitive structure and precision."""
    payload = {
        "cell": [[float(value).hex() for value in row] for row in cell],
        "scaled_positions": [[float(value).hex() for value in row] for row in scaled_positions],
        "numbers": [int(value) for value in atomic_numbers],
        "symprec": symprec.hex(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def primitive_data(atoms, symprec):
    if not isinstance(atoms, Atoms):
        raise TypeError("primitive_atoms must be an ASE Atoms object")
    if not np.all(atoms.pbc):
        raise ValueError("primitive cell must be periodic in all three directions")
    cell, positions, numbers, symprec = validate_primitive_arrays(
        atoms.cell.array, atoms.get_scaled_positions(wrap=False), atoms.numbers, symprec
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
    """One primitive motif site translated by an exact integer lattice vector."""

    site: int
    translation: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        if self.site < 0:
            raise ValueError("primitive site must be non-negative")
        if len(self.translation) != 3 or any(
            not isinstance(value, (int, np.integer)) for value in self.translation
        ):
            raise TypeError("lattice translation must contain exactly three integers")
        object.__setattr__(self, "site", int(self.site))
        object.__setattr__(self, "translation", tuple(int(value) for value in self.translation))
