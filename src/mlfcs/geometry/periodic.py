"""Minimum-image and tolerance searches in fully periodic three-dimensional lattices."""

from __future__ import annotations

from itertools import product
from math import prod

import numpy as np
from ase.geometry import minkowski_reduce

from mlfcs.foundation.arrays import require_allocation


class PeriodicGeometry:
    """Periodic-image geometry for a fixed three-dimensional lattice.

    Cartesian displacement vectors are compared modulo lattice translations.
    The class provides minimum-image reduction and exhaustive periodic-image
    matching within a Cartesian tolerance.

    A Minkowski-reduced basis is constructed once,

        cell_reduced = U @ cell,

    with integer unimodular ``U``. This changes only the lattice basis, not the
    periodic lattice, and enables bounded periodic-image searches.
    """

    def __init__(self, cell: object) -> None:
        """Construct periodic-search geometry for the given lattice."""
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
        """Return the minimum-image representative of each Cartesian displacement.

        For each displacement ``v``, find an integer lattice translation ``n``
        that minimizes ``||v + n @ cell||``. The returned image is the
        minimizing Cartesian displacement and the second return value is its
        Euclidean norm.

        Parameters
        ----------
        vectors
            Cartesian row vector of shape ``(3,)`` or an array of shape
            ``(n, 3)``.

        Returns
        -------
        images
            Minimum-image Cartesian displacements with the same vector shape
            as the input.
        lengths
            Corresponding Euclidean lengths. A single input returns a scalar.
        """
        images, lengths = self._nearest_reduced(vectors)
        if np.asarray(vectors).ndim == 1:
            return images[0], float(lengths[0])
        return images, lengths

    def _nearest_reduced(self, vectors: object) -> tuple[np.ndarray, np.ndarray]:
        """Find minimum images by testing the 27 neighboring reduced-cell translations.

        For a Minkowski-reduced three-dimensional lattice, the nearest image
        lies among the integer translations neighboring the wrapped
        reduced-coordinate representative.
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
        """Return all periodic images lying within a Cartesian tolerance.

        For each input displacement ``v_i``, find every integer lattice
        translation ``n`` satisfying

            ||v_i + n @ cell|| < tolerance.

        Parameters
        ----------
        vectors
            Finite Cartesian displacement rows of shape ``(n, 3)``.
        tolerance
            Positive Cartesian matching radius in the cell's length unit.

        Returns
        -------
        indices
            Input-vector index associated with each periodic match.
        shifts
            Integer original-cell translations satisfying the tolerance
            condition, shape ``(n_matches, 3)``.

        Notes
        -----
        Matching is performed in a Minkowski-reduced basis. A finite integer
        search box is derived from Cauchy-Schwarz bounds: for reduced row cell
        ``B``, the coordinate bound is

            abs(h_j + (v @ inv(B))_j) < tolerance * norm(inv(B)[:, j]).

        Every candidate is tested directly in Cartesian distance. Multiple
        matches are returned when present; uniqueness is the responsibility
        of the caller.
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
    """Require floating shift bounds to be safely representable as int64."""
    if (
        not np.all(np.isfinite(values))
        or np.any(values < -((1 << 63) - 1))
        or np.any(values >= (1 << 63))
    ):
        raise OverflowError(f"{name} cannot enter int64")


def _transform_image_shifts(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Transform integer image shifts from the reduced to the original lattice basis.

    If the reduced lattice satisfies ``cell_reduced = U @ cell``, then a
    reduced-basis image shift ``h`` corresponds to the original-cell shift

        n = h @ U.

    Parameters
    ----------
    left
        Reduced-basis integer shifts of shape ``(n, 3)``.
    right
        Integer reduction matrix ``U`` of shape ``(3, 3)``.

    Returns
    -------
    ndarray
        Original-cell integer shifts of shape ``(n, 3)``.

    Raises
    ------
    OverflowError
        If an exact transformed shift cannot be represented as int64.
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
