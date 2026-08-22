"""Periodic minimum-image geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from ase.geometry import find_mic, minkowski_reduce


@dataclass(frozen=True, slots=True)
class PeriodicGeometry:
    """One reduced-lattice implementation of periodic distance operations.

    The main calculation never enumerates a fixed image box in the supplied
    cell basis.  ASE selects a general minimum image; the reduced lattice is
    then used only to recover all images tied at that minimum.  This keeps
    skewed and unimodularly transformed frames on the same geometry rule.
    """

    cell: np.ndarray
    pbc: np.ndarray | bool = True
    atol: float = 1e-8
    rtol: float = 1e-10
    _reduction: np.ndarray = field(init=False, repr=False, compare=False)
    _closest_cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = field(
        init=False, repr=False, compare=False
    )
    _minimum_length_cache: dict[tuple[float, float, float], float] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        cell = np.asarray(self.cell, dtype=float)
        if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-12:
            raise ValueError("periodic geometry requires a nonsingular 3x3 cell")
        pbc = np.broadcast_to(np.asarray(self.pbc, dtype=bool), (3,)).copy()
        if not np.all(pbc):
            raise ValueError("MLFCS periodic geometry requires three periodic directions")
        _, reduction = minkowski_reduce(cell, pbc=pbc)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "pbc", pbc)
        object.__setattr__(self, "_reduction", np.rint(reduction).astype(np.int32))
        object.__setattr__(self, "_closest_cache", {})
        object.__setattr__(self, "_minimum_length_cache", {})

    def mic(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ASE's general minimum-image vectors and their lengths."""
        values = np.asarray(vectors, dtype=float)
        return find_mic(values, self.cell, pbc=self.pbc)

    def pair_distances(self, positions: np.ndarray) -> np.ndarray:
        """Return the complete general-MIC distance matrix for ``positions``."""
        values = np.asarray(positions, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("pair_distances expects an (n, 3) Cartesian array")
        _, lengths = self.mic((values[None, :, :] - values[:, None, :]).reshape((-1, 3)))
        return np.asarray(lengths).reshape((len(values), len(values)))

    def closest_images(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return all degenerate nearest images and their supercell shifts.

        Shifts are row-vector integer coefficients of ``cell``.  The first
        returned array contains Cartesian image vectors; the second contains
        the matching shifts.  The search is local in a Minkowski-reduced
        basis around ASE's MIC result, rather than in a fixed source-cell box.
        """
        value = np.asarray(vector, dtype=float)
        if value.shape != (3,):
            raise ValueError("closest_images expects one Cartesian 3-vector")
        key = tuple(float(component) for component in value)
        cached = self._closest_cache.get(key)
        if cached is not None:
            return cached
        mic, length = find_mic(value[None, :], self.cell, pbc=self.pbc)
        minimum = float(np.asarray(length).reshape(-1)[0])
        centre = np.rint((mic.reshape(3) - value) @ np.linalg.inv(self.cell)).astype(np.int32)
        local = np.asarray(tuple(product((-1, 0, 1), repeat=3)), dtype=np.int32)
        shifts = centre + local @ self._reduction
        images = value + shifts @ self.cell
        lengths = np.linalg.norm(images, axis=1)
        tied = np.abs(lengths - minimum) <= self.atol + self.rtol * max(minimum, 1.0)
        # A numerical tie may appear multiple times after a reduced-basis
        # transform.  Keep a deterministic unique representative of each
        # actual lattice shift.
        unique, locations = np.unique(shifts[tied], axis=0, return_index=True)
        order = np.argsort(locations)
        result = images[tied][locations[order]], unique[order]
        self._closest_cache[key] = result
        self._minimum_length_cache[key] = minimum
        return result

    def minimum_length(self, vector: np.ndarray) -> float:
        """Return a cached general-MIC length for one Cartesian vector."""
        value = np.asarray(vector, dtype=float)
        if value.shape != (3,):
            raise ValueError("minimum_length expects one Cartesian 3-vector")
        key = tuple(float(component) for component in value)
        cached = self._minimum_length_cache.get(key)
        if cached is not None:
            return cached
        _, length = self.mic(value[None, :])
        result = float(np.asarray(length).reshape(-1)[0])
        self._minimum_length_cache[key] = result
        return result

    def joint_closest_image_shifts(self, vectors: np.ndarray) -> np.ndarray:
        """Return mutually compatible nearest-image shifts for an anchored cluster.

        ``vectors`` contains the Cartesian vectors from the anchor to every
        tail atom before a supercell image is selected.  Every returned row
        chooses a nearest image for each tail and also requires every
        tail-to-tail vector to be a minimum image.  This is the joint image
        convention required by lattice-labelled cluster exports.
        """
        values = np.asarray(vectors, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("joint closest images expect an (n, 3) Cartesian array")
        if len(values) == 0:
            return np.zeros((1, 0, 3), dtype=np.int32)

        candidates: list[np.ndarray] = []
        for vector in values:
            _, shifts = self.closest_images(vector)
            candidates.append(shifts)

        compatible: list[np.ndarray] = []
        for selection in product(*candidates):
            shifts = np.asarray(selection, dtype=np.int32)
            images = values + shifts @ self.cell
            valid = True
            for left in range(len(images)):
                for right in range(left + 1, len(images)):
                    difference = images[right] - images[left]
                    length = float(np.linalg.norm(difference))
                    target = self.minimum_length(difference)
                    if abs(length - target) > self.atol + self.rtol * max(target, 1.0):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                compatible.append(shifts)
        if not compatible:
            return np.empty((0, len(values), 3), dtype=np.int32)
        return np.asarray(compatible, dtype=np.int32)


def unique_periodic_distances(
    values: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8
) -> list[float]:
    """Return sorted nonzero periodic distances with duplicates removed."""
    result: list[float] = []
    for value in np.sort(values):
        if value > atol and (not result or not np.isclose(value, result[-1], atol=atol, rtol=rtol)):
            result.append(float(value))
    if not result:
        raise ValueError("no periodic neighbors found; supercell is too small")
    return result


__all__ = ["PeriodicGeometry", "unique_periodic_distances"]
