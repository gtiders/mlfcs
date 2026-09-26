"""Periodic geometry with caller-declared physical precision."""

from __future__ import annotations

from itertools import product

import numpy as np
from ase.geometry import minkowski_reduce


class PeriodicGeometry:
    """Minimum-image geometry for one fully periodic lattice."""

    def __init__(self, cell: object) -> None:
        values = np.array(cell, dtype=np.float64, copy=True)
        if values.shape != (3, 3) or not np.all(np.isfinite(values)):
            raise ValueError("periodic geometry requires a finite cell with shape (3, 3)")
        if float(np.linalg.det(values)) == 0.0:
            raise ValueError("periodic geometry requires a nonsingular cell")
        reduced, transform = minkowski_reduce(values, pbc=True)
        self._cell = values
        self._cell.setflags(write=False)
        self._reduced = np.asarray(reduced, dtype=np.float64)
        self._reduced.setflags(write=False)
        self._inverse = np.linalg.inv(self._reduced)
        self._inverse.setflags(write=False)
        self._reduction = np.asarray(transform, dtype=np.int64)
        self._reduction.setflags(write=False)
        self._neighbors = np.asarray(tuple(product((-1, 0, 1), repeat=3)), dtype=np.int64)
        self._neighbors.setflags(write=False)

    @property
    def cell(self) -> np.ndarray:
        return self._cell

    def minimum_image(self, vectors: object) -> tuple[np.ndarray, np.ndarray | float]:
        """Return true minimum images for one or many Cartesian vectors."""
        values = np.asarray(vectors, dtype=np.float64)
        single = values.ndim == 1
        batch = np.atleast_2d(values)
        if batch.ndim != 2 or batch.shape[1] != 3 or not np.all(np.isfinite(batch)):
            raise ValueError("vectors must have shape (3,) or (n, 3)")
        if len(batch) == 0:
            return np.empty((0, 3)), np.empty(0)
        scaled = batch @ self._inverse
        base = -np.floor(scaled).astype(np.int64)
        shifts = base[:, None, :] + self._neighbors[None, :, :]
        candidates = batch[:, None, :] + shifts @ self._reduced
        distances = np.linalg.norm(candidates, axis=2)
        selected = np.argmin(distances, axis=1)
        images = candidates[np.arange(len(batch)), selected]
        lengths = distances[np.arange(len(batch)), selected]
        if single:
            return images[0], float(lengths[0])
        return images, lengths

    def closest_images(self, vector: object, *, symprec: float) -> tuple[np.ndarray, np.ndarray]:
        """Return all lattice images tied at the minimum within ``symprec`` angstrom."""
        _require_symprec(symprec)
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("vector must be a finite Cartesian triple")
        _, minimum_length = self.minimum_image(value)
        radius = minimum_length + symprec
        center = -(value @ self._inverse)
        span = radius * np.linalg.norm(self._inverse, axis=0)
        lower = np.ceil(np.nextafter(center - span, -np.inf)).astype(np.int64)
        upper = np.floor(np.nextafter(center + span, np.inf)).astype(np.int64)
        reduced_shifts = np.asarray(
            tuple(product(*(range(int(lo), int(hi) + 1) for lo, hi in zip(lower, upper)))),
            dtype=np.int64,
        ).reshape(-1, 3)
        images = value + reduced_shifts @ self._reduced
        tied = np.abs(np.linalg.norm(images, axis=1) - minimum_length) < symprec
        shifts = reduced_shifts[tied] @ self._reduction
        order = np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0]))
        return images[tied][order], shifts[order]


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
