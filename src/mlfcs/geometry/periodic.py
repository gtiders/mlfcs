"""Tolerance matching and nearest images in fully periodic 3D lattices."""

from __future__ import annotations

from itertools import product
from math import prod

import numpy as np
from ase.geometry import minkowski_reduce

from mlfcs._arrays import require_allocation


class PeriodicGeometry:
    """Periodic image searches in a fixed nonsingular three-dimensional cell.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
        Cartesian lattice vectors as rows, in angstrom.

    Notes
    -----
    Construction performs one Minkowski reduction. Readonly ``reduced``,
    ``inverse`` and integer ``reduction`` satisfy reduced = reduction @ cell.
    All displacement inputs are Cartesian row vectors. Searches do not mutate them.
    """

    def __init__(self, cell: object) -> None:
        """Validate the cell and cache its reduced basis, inverse and neighboring shifts."""
        values = np.asarray(cell, dtype=np.float64)
        if values.shape != (3, 3) or not np.all(np.isfinite(values)):
            raise ValueError("periodic geometry requires a finite cell with shape (3, 3)")
        if float(np.linalg.det(values)) == 0.0:
            raise ValueError("periodic geometry requires a nonsingular cell")
        reduced, transform = minkowski_reduce(values, pbc=True)
        self.reduced = np.asarray(reduced, dtype=np.float64)
        self.inverse = np.linalg.inv(self.reduced)
        self.reduction = np.asarray(transform, dtype=np.int64)
        for array in (self.reduced, self.inverse, self.reduction):
            array.setflags(write=False)
        self._neighbors = np.asarray(tuple(product((-1, 0, 1), repeat=3)), dtype=np.int64)
        self._neighbors.setflags(write=False)

    def minimum_image(self, vectors: object) -> tuple[np.ndarray, np.ndarray | float]:
        """Return nearest Cartesian images and their lengths, in the input length unit.

        ``vectors`` has shape (3,) or (n, 3). Outputs retain that vector shape;
        a single input returns a scalar length, a batch returns shape (n,).
        Equal-distance ties follow candidate enumeration order. Invalid inputs raise
        ValueError; nonrepresentable integer shifts raise OverflowError.
        """
        images, lengths = self._nearest_reduced(vectors)
        if np.asarray(vectors).ndim == 1:
            return images[0], float(lengths[0])
        return images, lengths

    def _nearest_reduced(self, vectors: object) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate 27 shifts around each reduced-cell displacement and select a minimum.

        Returns batched images (n, 3) and lengths (n,). Shift endpoints are checked
        before int64 conversion and before adding the neighboring offsets.
        """
        values = np.asarray(vectors, dtype=np.float64)
        batch = np.atleast_2d(values)
        if batch.ndim != 2 or batch.shape[1] != 3 or not np.all(np.isfinite(batch)):
            raise ValueError("vectors must have shape (3,) or (n, 3) and be finite")
        if not len(batch):
            return np.empty((0, 3)), np.empty(0)
        scaled = batch @ self.inverse
        if not np.all(np.isfinite(scaled)):
            raise OverflowError("periodic image shifts cannot enter int64")
        base_values = -np.floor(scaled)
        _require_int64_values(base_values, "periodic image shifts")
        if np.any(base_values <= -(1 << 63) + 1) or np.any(base_values >= (1 << 63) - 1):
            raise OverflowError("periodic image shifts cannot enter int64 safely")
        require_allocation("minimum-image candidates", (len(batch), 27, 3))
        # In the reduced basis, the nearest image is among these 27 neighbors.
        shifts = base_values.astype(np.int64)[:, None, :] + self._neighbors[None, :, :]
        candidates = batch[:, None, :] + shifts @ self.reduced
        distances = np.linalg.norm(candidates, axis=2)
        selected = np.argmin(distances, axis=1)
        images = candidates[np.arange(len(batch)), selected]
        lengths = distances[np.arange(len(batch)), selected]
        return images, lengths

    def matching_images(
        self, vectors: object, *, tolerance: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return all periodic matches strictly inside a Cartesian tolerance.

        Parameters
        ----------
        vectors : array_like, shape (n, 3)
            Finite Cartesian displacement rows, in the cell's length unit.
        tolerance : float
            Positive finite matching radius in that same unit.

        Returns
        -------
        indices : ndarray of int64, shape (n_matches,)
            Input vector index for each match; indices may repeat or be absent.
        shifts : ndarray of int64, shape (n_matches, 3)
            Original-cell shifts satisfying norm(vectors[indices] + shifts @ cell)
            < tolerance. Empty output has shape (0, 3).

        Notes
        -----
        For reduced row cell B, Cauchy-Schwarz bounds each integer coordinate by
        abs(h_j + (v @ inv(B))_j) < tolerance * norm(inv(B)[:, j]). Enumerate that
        box with outward-rounded endpoints and test distances directly. No unique
        match is assumed; callers enforce uniqueness where required.
        Invalid input raises ValueError and unrepresentable shifts raise OverflowError.
        """
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("vectors must have shape (n, 3) and be finite")
        center = -(values @ self.inverse)
        span = tolerance * np.linalg.norm(self.inverse, axis=0)
        # Expand floating endpoints outward; the Cartesian test below decides
        # strict tolerance membership after integer-box enumeration.
        lower_values = np.ceil(np.nextafter(center - span, -np.inf))
        upper_values = np.floor(np.nextafter(center + span, np.inf))
        _require_int64_values(lower_values, "periodic image shifts")
        _require_int64_values(upper_values, "periodic image shifts")
        lower, upper = lower_values.astype(np.int64), upper_values.astype(np.int64)
        candidate_indices, matched_shifts = [], []
        for index in np.flatnonzero(np.all(lower <= upper, axis=1)):
            extents = tuple(
                int(hi) - int(lo) + 1 for lo, hi in zip(lower[index], upper[index], strict=True)
            )
            require_allocation("periodic matching candidates", (prod(extents), 3))
            shifts = np.fromiter(
                (
                    coordinate
                    for shift in product(
                        *(
                            range(int(lo), int(hi) + 1)
                            for lo, hi in zip(lower[index], upper[index], strict=True)
                        )
                    )
                    for coordinate in shift
                ),
                dtype=np.int64,
            ).reshape(-1, 3)
            images = values[index] + shifts @ self.reduced
            matches = np.linalg.norm(images, axis=1) < tolerance
            original_shifts = _transform_image_shifts(shifts[matches], self.reduction)
            candidate_indices.extend([int(index)] * len(original_shifts))
            matched_shifts.extend(original_shifts)
        return np.asarray(candidate_indices, dtype=np.int64), np.asarray(
            matched_shifts, dtype=np.int64
        ).reshape(-1, 3)


def _require_int64_values(values: np.ndarray, name: str) -> None:
    """Check finite float shift bounds against the int64 conversion endpoints."""
    if (
        not np.all(np.isfinite(values))
        or np.any(values < -((1 << 63) - 1))
        or np.any(values >= (1 << 63))
    ):
        raise OverflowError(f"{name} cannot enter int64")


def _transform_image_shifts(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Transform image shifts from the reduced basis to the original cell basis.

    ``left`` has shape (n, 3) and ``right`` is the (3, 3) lattice reduction.
    Both contain integers. Return an int64 array of shape (n, 3), with rows
    equal to left @ right. Python sums are evaluated before dtype conversion;
    results outside [-INT64_MAX, INT64_MAX] raise OverflowError. Storage limits
    also raise OverflowError. Inputs are not modified.
    """
    require_allocation("periodic image shift product", (len(left), right.shape[1]))
    result = np.empty((len(left), right.shape[1]), dtype=np.int64)
    for i, row in enumerate(left):
        for j in range(right.shape[1]):
            total = sum(int(row[k]) * int(right[k, j]) for k in range(right.shape[0]))
            if not -(1 << 63) < total < (1 << 63):
                raise OverflowError("periodic image shifts cannot enter int64")
            result[i, j] = total
    return result


__all__ = ["PeriodicGeometry"]
