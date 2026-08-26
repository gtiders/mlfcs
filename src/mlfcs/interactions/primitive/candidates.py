"""Primitive cutoff resolution and orbit-seed candidate generation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from ase import Atoms
from ase.geometry import minkowski_reduce
from ase.neighborlist import neighbor_list

from mlfcs.interactions.keys import InteractionKey
from mlfcs.structure.periodic_geometry import unique_periodic_distances


def resolve_primitive_cutoff(
    primitive: Atoms,
    cutoff: float | None,
    *,
    reference: Atoms | None = None,
) -> float:
    """Resolve a distance, neighbor shell, or finite-reference cutoff."""
    if cutoff is None:
        if reference is None:
            raise ValueError("cutoff=None requires an explicit reference supercell")
        reduced_cell, _operation = minkowski_reduce(reference.cell, pbc=reference.pbc)
        periodic_lengths = np.linalg.norm(
            np.asarray(reduced_cell)[np.asarray(reference.pbc)], axis=1
        )
        if not len(periodic_lengths):
            raise ValueError("cutoff=None requires at least one periodic direction")
        upper = float(np.min(periodic_lengths)) + 1e-8
        first, second, shifts, distances = neighbor_list(
            "ijSd", reference, upper, self_interaction=False
        )
        by_pair: dict[tuple[int, int], list[tuple[float, tuple[int, int, int]]]] = {}
        for atom_i, atom_j, shift, distance in zip(
            first, second, shifts, distances, strict=True
        ):
            by_pair.setdefault((int(atom_i), int(atom_j)), []).append(
                (float(distance), tuple(int(value) for value in shift))
            )
        boundaries = []
        for (atom_i, atom_j), images in by_pair.items():
            ordered = sorted(set(images))
            if atom_i == atom_j and ordered:
                boundaries.append(ordered[0][0])
            elif atom_i != atom_j and len(ordered) > 1:
                boundaries.append(ordered[1][0])
        if not boundaries:
            raise RuntimeError("could not determine a finite-cell cutoff boundary")
        resolved = min(boundaries) - 0.01
        if resolved <= 0:
            raise ValueError("reference supercell is too small for cutoff=None")
        return float(resolved)

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
        first, _second, distances = neighbor_list(
            "ijd", primitive, radius, self_interaction=False
        )
        shells = []
        for site in range(len(primitive)):
            try:
                shells.append(unique_periodic_distances(distances[first == site]))
            except ValueError:
                shells.append([])
        if all(len(values) > shell for values in shells):
            return float(
                max((values[shell - 1] + values[shell]) / 2.0 for values in shells)
            )
        radius *= 2.0
    raise RuntimeError("could not resolve the requested primitive neighbor shell")


def _primitive_neighbors(primitive: Atoms, cutoff: float):
    first, second, shifts, distances = neighbor_list(
        "ijSd", primitive, cutoff, self_interaction=True
    )
    result: list[list[tuple[int, int, int, int]]] = [[] for _ in primitive]
    for anchor, site, shift, distance in zip(
        first, second, shifts, distances, strict=True
    ):
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
            if all(
                np.linalg.norm(point - coordinate(previous)) < cutoff
                for previous in prefix
            ):
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
