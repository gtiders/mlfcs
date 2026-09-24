"""Immutable primitive-cell data used by the rewritten core."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms


def _readonly(values: object, *, dtype: object) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PrimitiveCell:
    """A primitive crystal motif without any supercell or training state.

    Coordinates use ASE's row-vector convention: a fractional position ``x``
    has Cartesian position ``x @ cell``.  The object owns read-only copies of
    all arrays so downstream algebra cannot mutate its identity accidentally.
    """

    cell: np.ndarray
    scaled_positions: np.ndarray
    numbers: np.ndarray
    symprec: float

    def __post_init__(self) -> None:
        cell = _readonly(self.cell, dtype=np.float64)
        positions = np.asarray(self.scaled_positions, dtype=np.float64)
        numbers = np.asarray(self.numbers)
        symprec = float(self.symprec)
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
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "scaled_positions", _readonly(wrapped, dtype=np.float64))
        object.__setattr__(self, "numbers", _readonly(numbers, dtype=np.int32))
        object.__setattr__(self, "symprec", symprec)

    @classmethod
    def from_atoms(cls, atoms: Atoms, *, symprec: float) -> PrimitiveCell:
        """Copy and certify one fully periodic primitive ASE structure."""
        if not bool(np.all(atoms.pbc)):
            raise ValueError("primitive cell must be periodic in all three directions")
        return cls(
            cell=np.asarray(atoms.cell),
            scaled_positions=atoms.get_scaled_positions(wrap=False),
            numbers=atoms.numbers,
            symprec=symprec,
        )

    @property
    def size(self) -> int:
        return len(self.numbers)

    @property
    def fingerprint(self) -> str:
        """Stable identity of the declared primitive structure and precision."""
        payload = {
            "cell": [[float(value).hex() for value in row] for row in self.cell],
            "scaled_positions": [
                [float(value).hex() for value in row] for row in self.scaled_positions
            ],
            "numbers": [int(value) for value in self.numbers],
            "symprec": self.symprec.hex(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def cartesian_positions(self) -> np.ndarray:
        return self.scaled_positions @ self.cell

    def to_atoms(self) -> Atoms:
        """Return a detached ASE representation."""
        return Atoms(
            numbers=self.numbers,
            scaled_positions=self.scaled_positions,
            cell=self.cell,
            pbc=True,
        )
