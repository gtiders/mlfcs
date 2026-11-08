"""Lattice sites and periodic-image geometry in three dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np
from ase.geometry import minkowski_reduce

from mlfcs.foundation.arrays import require_allocation


@dataclass(frozen=True, order=True, slots=True)
class LatticeSite:
    """Unique address for one atom in an infinitely periodic crystal.

    A lattice site is specified by ``(site, translation)``:
    - ``site``: index of the atom inside the primitive unit cell (i)
    - ``translation``: integer lattice vector ``n = (tx, ty, tz)``,
      counting how many primitive cell steps to translate.

    With row-wise lattice basis vectors, its Cartesian position is:
        r(i, n) = (s_i + n) @ cell
    where s_i is the fractional coordinate of atom i inside the primitive cell.

    Important: ``translation`` is an integer lattice offset, **not**
    a Cartesian displacement in real space.
    """

    site: int
    translation: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        """Validate and normalize lattice-site address."""
        if self.site < 0:
            raise ValueError("primitive site must be non-negative")
        if len(self.translation) != 3 or any(
            not isinstance(value, (int, np.integer)) for value in self.translation
        ):
            raise TypeError("lattice translation must contain exactly three integers")
        object.__setattr__(self, "site", int(self.site))
        object.__setattr__(self, "translation", tuple(int(value) for value in self.translation))


def match_primitive_site(
    cell,
    scaled_positions,
    atomic_numbers,
    position,
    species,
    *,
    tolerance,
    geometry=None,
    context="",
):
    """Locate the unique motif site that a fractional position belongs to.

    Return ``(site, translation)`` satisfying

        position = scaled_positions[site] + translation

    where ``translation`` is the integer lattice shift **added to the motif
    position**, i.e. the ``LatticeSite`` convention. ``position`` is a
    fractional row coordinate that need not lie inside the reference cell;
    only motif sites carrying ``species`` are candidates. This function owns
    both the uniqueness rule and the sign convention, so callers never negate
    a periodic-image shift themselves.

    Parameters
    ----------
    cell
        Lattice vectors as rows, shape ``(3, 3)``, in angstrom.
    scaled_positions
        Fractional motif coordinates of shape ``(n_atoms, 3)``.
    atomic_numbers
        Motif species, one entry per row of ``scaled_positions``.
    position
        Fractional row coordinate of shape ``(3,)``.
    species
        Species the matched motif site must carry.
    tolerance
        Positive Cartesian matching radius in angstrom.
    geometry
        Optional prepared ``Lattice`` for ``cell``; supplying it avoids
        rebuilding the Minkowski reduction at every call.
    context
        Optional label prefixed to the raised message.

    Raises
    ------
    ValueError
        If no candidate of ``species`` exists, or if the position matches no
        candidate or more than one within ``tolerance``.
    """
    geometry = Lattice(cell) if geometry is None else geometry
    candidates = np.flatnonzero(np.asarray(atomic_numbers) == species)
    prefix = f"{context}: " if context else ""
    if not len(candidates):
        raise ValueError(f"{prefix}no primitive site carries species {species}")
    differences = (np.asarray(position, dtype=np.float64) - scaled_positions[candidates]) @ cell
    matches, shifts = geometry.matching_images(differences, tolerance=tolerance)
    if len(matches) != 1:
        raise ValueError(
            f"{prefix}position matches {len(matches)} primitive sites "
            f"within tolerance {tolerance:g} angstrom"
        )
    return int(candidates[matches[0]]), tuple(-int(value) for value in shifts[0])


class Lattice:
    """Periodic-image geometry of a fixed three-dimensional lattice.

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

    def lattice_combination(
        self, vector: object, *, tolerance: float, context: str = ""
    ) -> tuple[int, int, int]:
        """Return the integer lattice combination that realizes a Cartesian vector.

        For row lattice vectors, ``vector = n @ cell`` with integer ``n``. The
        unique admissible combination is returned as a plain integer tuple; it
        is the inverse of the periodic-image convention used by
        :meth:`matching_images`, so this method is the single owner of that sign.

        Parameters
        ----------
        vector
            Cartesian row vector of shape ``(3,)``.
        tolerance
            Positive Cartesian matching radius in the cell's length unit.
        context
            Optional label prefixed to the error message by the caller.

        Raises
        ------
        ValueError
            If the vector matches no lattice point, or more than one, within
            ``tolerance``.
        """
        matches, shifts = self.matching_images(
            np.atleast_2d(np.asarray(vector, dtype=np.float64)), tolerance=tolerance
        )
        if len(matches) != 1:
            prefix = f"{context}: " if context else ""
            raise ValueError(
                f"{prefix}lattice vector matches {len(matches)} lattice points "
                f"within tolerance {tolerance:g}"
            )
        return tuple(-int(value) for value in shifts[0])


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


__all__ = ["Lattice", "LatticeSite", "match_primitive_site"]
