"""Enumerate periodic neighbors and geometrically admissible force-constant clusters.

Candidate clusters are defined by a force-constant order, an all-pairs
distance cutoff, and a maximum body order. Enumeration uses separate count
and fill passes to allow exact array allocation.
"""

import math
import operator

import numpy as np
from numba import njit

from mlfcs.foundation.arrays import INTP_MAX, readonly, require_allocation, require_bound


def neighbor_translation_bounds(cell, cutoff):
    """Return periodic-image search bounds sufficient for the given cutoff.

    The bounds are integer half-widths of lattice translations along each
    cell direction and conservatively include every periodic image that may
    lie within the Cartesian cutoff. Nonfinite or unrepresentable widths
    raise OverflowError.
    """
    radii = cutoff * np.linalg.norm(np.linalg.inv(cell), axis=0) + 1.0
    if not np.all(np.isfinite(radii)):
        raise OverflowError("neighbor translation bounds are not finite")
    bounds = tuple(math.ceil(float(v)) for v in radii)
    for value in bounds:
        require_bound("neighbor traversal width", 2 * value + 1)
    return bounds


def enumerate_candidate_labels(cell, scaled_positions, *, order, cutoff, max_body_order):
    """Enumerate force-constant cluster labels satisfying the geometric truncation.

    Each cluster contains ``order`` lattice sites, with the first site fixed
    in the reference cell. Every pair of sites must lie within ``cutoff``,
    and the number of distinct atomic positions must not exceed
    ``max_body_order``. Returns integer labels of shape ``(n_clusters,
    order, 4)``, one ``(site, tx, ty, tz)`` tuple per tensor index; a
    truncation whose labels cannot be represented raises OverflowError.
    """
    order, max_body_order = operator.index(order), operator.index(max_body_order)
    cutoff = float(cutoff)
    if order < 2 or not 1 <= max_body_order <= order:
        raise ValueError("order must be >= 2 and max_body_order must lie in 1..order")
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be a positive finite distance")
    cell = readonly(cell, np.float64)
    scaled_positions = readonly(scaled_positions, np.float64)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("cell must be a finite 3x3 matrix")
    if scaled_positions.ndim != 2 or scaled_positions.shape[1] != 3:
        raise ValueError("scaled positions must have shape (N, 3)")
    if not np.all(np.isfinite(scaled_positions)) or np.any(
        (scaled_positions < 0) | (scaled_positions >= 1)
    ):
        raise ValueError("scaled positions must be finite and wrapped into [0, 1)")
    require_allocation("candidate stack", (order - 1,))
    bounds = np.asarray(neighbor_translation_bounds(cell, cutoff), dtype=np.int64)
    require_allocation("neighbor offsets", (len(scaled_positions) + 1,))
    offsets = np.zeros(len(scaled_positions) + 1, dtype=np.int64)
    count = _count_neighbors(
        scaled_positions,
        cell,
        bounds,
        cutoff,
        offsets,
        np.empty((0, 4), dtype=np.int64),
        np.empty((0, 3)),
        INTP_MAX // 32,
    )
    if count < 0:
        raise OverflowError("actual neighbor labels exceed representable allocation capacity")
    require_allocation("neighbor labels", (count, 4))
    require_allocation("neighbor points", (count, 3))
    labels = np.empty((count, 4), dtype=np.int64)
    points = np.empty((count, 3), dtype=np.float64)
    _fill_neighbors(scaled_positions, cell, bounds, cutoff, offsets, labels, points)
    counts = np.zeros(len(scaled_positions), dtype=np.int64)
    output = np.empty((0, order, 4), dtype=np.int64)
    count = _count_candidates(
        labels,
        offsets,
        points,
        order,
        max_body_order,
        cutoff,
        counts,
        output,
        INTP_MAX // (order * 4 * 8),
    )
    if count < 0:
        raise OverflowError("actual candidate labels exceed representable allocation capacity")
    require_allocation("candidate labels", (count, order, 4))
    output = np.empty((count, order, 4), dtype=np.int64)
    _fill_candidates(labels, offsets, points, order, max_body_order, cutoff, counts, output)
    return output


@njit(cache=True, inline="always")
def _neighbors(positions, cell, bounds, cutoff, offsets, labels, points, fill, limit=0):
    """Count or store the periodic neighbors of each anchor within the cutoff.

    One traversal run twice: the sizing pass returns the total and writes
    ``offsets``, the fill pass relies on it. Inputs are wrapped fractional
    positions with row cell vectors; labels are ``(site, tx, ty, tz)``
    relative to the anchor.
    """
    total = 0
    for anchor in range(len(positions)):
        offsets[anchor] = total
        for site in range(len(positions)):
            for x in range(-bounds[0], bounds[0] + 1):
                for y in range(-bounds[1], bounds[1] + 1):
                    for z in range(-bounds[2], bounds[2] + 1):
                        shift = (x, y, z)
                        vector = np.zeros(3)
                        point = np.zeros(3)
                        for j in range(3):
                            for k in range(3):
                                vector[j] += (
                                    positions[site, k] + shift[k] - positions[anchor, k]
                                ) * cell[k, j]
                                point[j] += (positions[site, k] + shift[k]) * cell[k, j]
                        distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
                        if distance < cutoff:
                            if not fill and total == limit:
                                return -1
                            if fill:
                                labels[total, 0] = site
                                labels[total, 1:] = np.array(shift)
                                points[total] = point
                            total += 1
        offsets[anchor + 1] = total
    return total


@njit(cache=True, inline="always")
def _candidates(labels, offsets, points, order, body_order, cutoff, counts, output, fill, limit=0):
    """Enumerate the admissible cluster combinations of one anchor.

    Sites combine nondecreasing with repetition, so repeated lattice sites
    never generate permutation duplicates; each new site must lie within
    ``cutoff`` of all selected ones, and the distinct-position count may
    not exceed ``body_order``. Sizing and filling share this traversal.
    """
    total = 0
    for anchor in range(len(offsets) - 1):
        selected = np.zeros(order - 1, dtype=np.int64)
        next_index = np.zeros(order - 1, dtype=np.int64)
        depth = 0
        next_index[0] = offsets[anchor]
        count = 0
        while depth >= 0:
            index = next_index[depth]
            if index >= offsets[anchor + 1]:
                depth -= 1
                continue
            next_index[depth] += 1
            connected = True
            for earlier in range(depth):
                distance = 0.0
                for j in range(3):
                    value = points[index, j] - points[selected[earlier], j]
                    distance += value * value
                if math.sqrt(distance) >= cutoff:
                    connected = False
                    break
            if not connected:
                continue
            selected[depth] = index
            bodies = 1  # The anchored primitive site.
            for k in range(depth + 1):
                label = labels[selected[k]]
                unique = not (
                    label[0] == anchor and label[1] == 0 and label[2] == 0 and label[3] == 0
                )
                for earlier in range(k):
                    if np.all(label == labels[selected[earlier]]):
                        unique = False
                        break
                if unique:
                    bodies += 1
            if bodies > body_order:
                continue
            if depth == order - 2:
                if not fill and total == limit:
                    return -1
                if fill:
                    output[total, 0, 0] = anchor
                    output[total, 0, 1:] = 0
                    for k in range(order - 1):
                        output[total, k + 1] = labels[selected[k]]
                total += 1
                count += 1
            else:
                depth += 1
                next_index[depth] = index  # Nondecreasing combinations with repetition.
        counts[anchor] = count
    return total


@njit(cache=True)
def _count_neighbors(positions, cell, bounds, cutoff, offsets, labels, points, limit):
    """Count periodic neighbors and build anchor offsets."""
    return _neighbors(positions, cell, bounds, cutoff, offsets, labels, points, False, limit)


@njit(cache=True)
def _fill_neighbors(positions, cell, bounds, cutoff, offsets, labels, points):
    """Fill preallocated periodic-neighbor buffers."""
    return _neighbors(positions, cell, bounds, cutoff, offsets, labels, points, True, 0)


def _count_candidates(labels, offsets, points, order, body_order, cutoff, counts, output, limit):
    """Count admissible candidate clusters for exact output allocation."""
    return _candidates(
        labels, offsets, points, order, body_order, cutoff, counts, output, False, limit
    )


@njit(cache=True)
def _fill_candidates(labels, offsets, points, order, body_order, cutoff, counts, output):
    """Fill the preallocated candidate-cluster buffer."""
    return _candidates(labels, offsets, points, order, body_order, cutoff, counts, output, True, 0)
