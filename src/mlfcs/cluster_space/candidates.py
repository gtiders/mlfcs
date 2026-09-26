"""Primitive candidates determined only by physical geometry inputs."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from ase.neighborlist import neighbor_list

from mlfcs.cluster_space.models import Cluster
from mlfcs.core import LatticeSite, PrimitiveCell
from mlfcs.core.geometry import unique_distances


def resolve_cutoff(primitive: PrimitiveCell, cutoff: float) -> float:
    """Resolve a positive radius or negative one-based neighbor-shell count."""
    value = float(cutoff)
    if not np.isfinite(value) or value == 0.0:
        raise ValueError("cutoff must be a positive length or a negative shell count")
    if value > 0.0:
        return value
    shells = int(-value)
    if shells < 1 or value != -shells:
        raise ValueError("a negative cutoff must be an integer neighbor-shell count")

    atoms = primitive.to_atoms()
    radius = max(float(np.min(np.linalg.norm(primitive.cell, axis=1))), 1.0)
    while True:
        first, _second, distances = neighbor_list("ijd", atoms, radius, self_interaction=False)
        per_site = [
            unique_distances(distances[first == site], symprec=primitive.symprec)
            for site in range(primitive.size)
        ]
        if all(len(values) > shells for values in per_site):
            boundaries = [(values[shells - 1] + values[shells]) / 2.0 for values in per_site]
            return float(max(boundaries))
        radius *= 2.0
        if not np.isfinite(radius):
            raise ValueError(f"could not resolve neighbor shell {shells}")


def _neighbors(primitive: PrimitiveCell, cutoff: float) -> tuple[tuple[LatticeSite, ...], ...]:
    first, second, shifts, distances = neighbor_list(
        "ijSd", primitive.to_atoms(), cutoff, self_interaction=True
    )
    result: list[set[LatticeSite]] = [set() for _ in range(primitive.size)]
    for anchor, site, shift, distance in zip(first, second, shifts, distances, strict=True):
        if float(distance) < cutoff:
            result[int(anchor)].add(LatticeSite(int(site), tuple(int(value) for value in shift)))
    return tuple(tuple(sorted(values)) for values in result)


def iter_candidates(
    primitive: PrimitiveCell,
    *,
    order: int,
    cutoff: float,
    max_body_order: int,
) -> Iterator[Cluster]:
    """Yield pairwise-connected anchored clusters in deterministic order."""
    if order < 2:
        raise ValueError("cluster order must be at least two")
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be a positive finite distance in angstrom")
    if not 1 <= max_body_order <= order:
        raise ValueError("max_body_order must lie between one and the cluster order")
    neighbors = _neighbors(primitive, cutoff)
    positions = primitive.scaled_positions
    cell = primitive.cell

    def point(label: LatticeSite) -> np.ndarray:
        return (positions[label.site] + np.asarray(label.translation)) @ cell

    for anchor in range(primitive.size):
        choices = neighbors[anchor]
        prefix: list[LatticeSite] = []

        def extend(
            start: int,
            *,
            choices: tuple[LatticeSite, ...] = choices,
            prefix: list[LatticeSite] = prefix,
        ) -> Iterator[tuple[LatticeSite, ...]]:
            if len(prefix) == order - 1:
                yield tuple(prefix)
                return
            for location in range(start, len(choices)):
                candidate = choices[location]
                coordinate = point(candidate)
                if all(
                    np.linalg.norm(coordinate - point(previous)) < cutoff for previous in prefix
                ):
                    prefix.append(candidate)
                    yield from extend(location)
                    prefix.pop()

        origin = LatticeSite(anchor)
        for tail in extend(0):
            cluster = Cluster((origin, *tail))
            if cluster.body_order <= max_body_order:
                yield cluster


__all__ = ["iter_candidates", "resolve_cutoff"]
