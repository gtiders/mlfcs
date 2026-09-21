"""Primitive cutoff resolution and orbit-seed candidate generation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from mlfcs.interactions.keys import InteractionKey
from mlfcs.structure.periodic_geometry import unique_periodic_distances


def resolve_primitive_cutoff(primitive: Atoms, cutoff: float) -> float:
    """Resolve an explicit interaction radius.

    A positive value is a distance in angstrom and a negative integer is a primitive
    neighbour-shell index.  ``None`` is rejected: a primitive interaction model has to be
    fixed by the primitive structure and an explicit radius, never by the size of the
    finite reference that happens to observe it.  Whether a reference can identify that
    model is a separate question, answered by realization identifiability, which raises
    ``InteractionAliasingError`` instead of quietly shortening the model.
    """
    if cutoff is None:
        raise ValueError(
            "cutoff must be a positive distance in angstrom or a negative neighbour-shell "
            "index; the reference-resolved cutoff=None is not supported. Pick the radius of "
            "the primitive model explicitly and let realization identifiability check the "
            "reference."
        )
    value = float(cutoff)
    if value > 0:
        return value
    if not value.is_integer():
        raise ValueError("cutoff must be a positive distance or negative integer shell")
    shell = -int(value)
    if shell < 1:
        raise ValueError("neighbor shell must be positive")
    radius = max(float(np.min(np.linalg.norm(np.asarray(primitive.cell), axis=1))), 1.0)
    for _ in range(16):
        first, _second, distances = neighbor_list("ijd", primitive, radius, self_interaction=False)
        shells = []
        for site in range(len(primitive)):
            try:
                shells.append(unique_periodic_distances(distances[first == site]))
            except ValueError:
                shells.append([])
        if all(len(values) > shell for values in shells):
            return float(max((values[shell - 1] + values[shell]) / 2.0 for values in shells))
        radius *= 2.0
    raise RuntimeError("could not resolve the requested primitive neighbor shell")


def _primitive_neighbors(primitive: Atoms, cutoff: float):
    first, second, shifts, distances = neighbor_list(
        "ijSd", primitive, cutoff, self_interaction=True
    )
    result: list[list[tuple[int, int, int, int]]] = [[] for _ in primitive]
    for anchor, site, shift, distance in zip(first, second, shifts, distances, strict=True):
        if float(distance) < cutoff:
            result[int(anchor)].append((int(site), *(int(value) for value in shift)))
    return [tuple(sorted(set(values))) for values in result]


def _compatible_tails(candidates, length, primitive: Atoms, cutoff: float):
    positions = primitive.get_scaled_positions(wrap=False)
    cell = np.asarray(primitive.cell)
    prefix: list[tuple[int, int, int, int]] = []

    def coordinate(label):
        return (positions[label[0]] + np.asarray(label[1:], dtype=float)) @ cell

    def extend(start):
        if len(prefix) == length:
            yield tuple(prefix)
            return
        for location in range(start, len(candidates)):
            candidate = candidates[location]
            point = coordinate(candidate)
            if all(np.linalg.norm(point - coordinate(previous)) < cutoff for previous in prefix):
                prefix.append(candidate)
                yield from extend(location)
                prefix.pop()

    yield from extend(0)


def iter_primitive_candidates(
    primitive: Atoms,
    *,
    radius: float,
    order: int,
    max_body_order: int | None,
) -> Iterator[InteractionKey]:
    """Yield anchored, body-order-filtered seeds without symmetry expansion."""
    neighbors = _primitive_neighbors(primitive, radius)
    for anchor in range(len(primitive)):
        for tail in _compatible_tails(neighbors[anchor], order - 1, primitive, radius):
            key = InteractionKey.from_labels(((anchor, 0, 0, 0), *tail))
            if max_body_order is None or len(set(key.labels)) <= max_body_order:
                yield key


__all__ = ["iter_primitive_candidates", "resolve_primitive_cutoff"]
