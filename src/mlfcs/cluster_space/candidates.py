"""Two-pass enumeration with bounded sizing and unchecked fill loops."""

import math
import operator

import numpy as np
from numba import njit

from mlfcs._arrays import INTP_MAX, readonly, require_allocation, require_bound


def neighbor_translation_bounds(cell, cutoff):
    radii = cutoff * np.linalg.norm(np.linalg.inv(cell), axis=0) + 1.0
    if not np.all(np.isfinite(radii)):
        raise OverflowError("neighbor translation bounds are not finite")
    bounds = tuple(math.ceil(float(v)) for v in radii)
    for value in bounds:
        require_bound("neighbor traversal width", 2 * value + 1)
    return bounds


def _candidate_labels(cell, scaled_positions, *, order, cutoff, max_body_order):
    """Validate and size actual outputs immediately before their fill passes."""
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
    count = neighbors(
        scaled_positions,
        cell,
        bounds,
        cutoff,
        offsets,
        np.empty((0, 4), dtype=np.int64),
        np.empty((0, 3)),
        False,
        INTP_MAX // 32,
    )
    if count < 0:
        raise OverflowError("actual neighbor labels exceed representable allocation capacity")
    require_allocation("neighbor labels", (count, 4))
    require_allocation("neighbor points", (count, 3))
    labels = np.empty((count, 4), dtype=np.int64)
    points = np.empty((count, 3), dtype=np.float64)
    neighbors(scaled_positions, cell, bounds, cutoff, offsets, labels, points, True)
    counts = np.zeros(len(scaled_positions), dtype=np.int64)
    output = np.empty((0, order, 4), dtype=np.int64)
    count = candidates(
        labels,
        offsets,
        points,
        order,
        max_body_order,
        cutoff,
        counts,
        output,
        False,
        INTP_MAX // (order * 4 * 8),
    )
    if count < 0:
        raise OverflowError("actual candidate labels exceed representable allocation capacity")
    require_allocation("candidate labels", (count, order, 4))
    output = np.empty((count, order, 4), dtype=np.int64)
    candidates(labels, offsets, points, order, max_body_order, cutoff, counts, output, True)
    return output


@njit(cache=True, inline="always")
def _neighbors(positions, cell, bounds, cutoff, offsets, labels, points, fill, limit=0):
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
    return _neighbors(positions, cell, bounds, cutoff, offsets, labels, points, False, limit)


@njit(cache=True)
def _fill_neighbors(positions, cell, bounds, cutoff, offsets, labels, points):
    return _neighbors(positions, cell, bounds, cutoff, offsets, labels, points, True, 0)


def neighbors(positions, cell, bounds, cutoff, offsets, labels, points, fill, limit=0):
    """Select the sizing/fill entry once, outside the compiled traversal."""
    if fill:
        return _fill_neighbors(positions, cell, bounds, cutoff, offsets, labels, points)
    return _count_neighbors(positions, cell, bounds, cutoff, offsets, labels, points, limit)


@njit(cache=True)
def _count_candidates(labels, offsets, points, order, body_order, cutoff, counts, output, limit):
    return _candidates(
        labels, offsets, points, order, body_order, cutoff, counts, output, False, limit
    )


@njit(cache=True)
def _fill_candidates(labels, offsets, points, order, body_order, cutoff, counts, output):
    return _candidates(labels, offsets, points, order, body_order, cutoff, counts, output, True, 0)


def candidates(labels, offsets, points, order, body_order, cutoff, counts, output, fill, limit=0):
    """One shared algorithm, compiled with a constant mode at each entry."""
    if fill:
        return _fill_candidates(labels, offsets, points, order, body_order, cutoff, counts, output)
    return _count_candidates(
        labels, offsets, points, order, body_order, cutoff, counts, output, limit
    )
