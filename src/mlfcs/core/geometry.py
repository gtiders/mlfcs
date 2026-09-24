"""Periodic geometry with caller-declared physical precision."""

from __future__ import annotations

from itertools import product

import numpy as np
from ase.geometry import minkowski_reduce
from ase.geometry.geometry import general_find_mic


class PeriodicGeometry:
    """Minimum-image geometry for one fully periodic lattice."""

    def __init__(self, cell: object) -> None:
        values = np.array(cell, dtype=np.float64, copy=True)
        if values.shape != (3, 3) or not np.all(np.isfinite(values)):
            raise ValueError("periodic geometry requires a finite cell with shape (3, 3)")
        if float(np.linalg.det(values)) == 0.0:
            raise ValueError("periodic geometry requires a nonsingular cell")
        _reduced, transform = minkowski_reduce(values, pbc=True)
        self._cell = values
        self._cell.setflags(write=False)
        self._reduction = np.rint(transform).astype(np.int64)
        self._reduction.setflags(write=False)

    @property
    def cell(self) -> np.ndarray:
        return self._cell

    def minimum_image(self, vectors: object) -> tuple[np.ndarray, np.ndarray | float]:
        """Return true minimum images for one or many Cartesian vectors."""
        values = np.asarray(vectors, dtype=np.float64)
        single = values.ndim == 1
        batch = np.atleast_2d(values)
        if batch.ndim != 2 or batch.shape[1] != 3:
            raise ValueError("vectors must have shape (3,) or (n, 3)")
        images, lengths = general_find_mic(batch, self.cell, pbc=np.ones(3, dtype=bool))
        if single:
            return images[0], float(lengths[0])
        return images, np.asarray(lengths)

    def closest_images(self, vector: object, *, symprec: float) -> tuple[np.ndarray, np.ndarray]:
        """Return all lattice images tied at the minimum within ``symprec`` angstrom."""
        _require_symprec(symprec)
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (3,):
            raise ValueError("vector must have shape (3,)")
        minimum_vector, minimum_length = self.minimum_image(value)
        center = np.rint((minimum_vector - value) @ np.linalg.inv(self.cell)).astype(np.int64)
        local = np.asarray(tuple(product((-1, 0, 1), repeat=3)), dtype=np.int64)
        shifts = center + local @ self._reduction
        images = value + shifts @ self.cell
        tied = np.abs(np.linalg.norm(images, axis=1) - minimum_length) < symprec
        unique_shifts, locations = np.unique(shifts[tied], axis=0, return_index=True)
        order = np.argsort(locations)
        return images[tied][locations[order]], unique_shifts[order]


def unique_distances(values: object, *, symprec: float) -> tuple[float, ...]:
    """Group nonzero Cartesian distances using the declared length precision."""
    _require_symprec(symprec)
    distances = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distances must be finite")
    result: list[float] = []
    for value in np.sort(distances):
        if value >= symprec and (not result or value - result[-1] >= symprec):
            result.append(float(value))
    return tuple(result)


def _require_symprec(symprec: float) -> None:
    if not np.isfinite(symprec) or symprec <= 0.0:
        raise ValueError("symprec must be a positive finite length in angstrom")


__all__ = ["PeriodicGeometry", "unique_distances"]
