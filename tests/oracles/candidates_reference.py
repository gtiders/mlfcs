"""Primitive candidates determined only by physical geometry inputs."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from ase.neighborlist import neighbor_list

from mlfcs import ClusterSpace
from mlfcs.cluster_space.model import Cluster
from mlfcs.geometry.primitive import LatticeSite


def _neighbors(primitive: ClusterSpace, cutoff: float) -> tuple[tuple[LatticeSite, ...], ...]:
    """List periodic neighbors strictly within cutoff of each primitive motif site.

    ``primitive`` supplies row lattice vectors and wrapped fractional positions;
    ``cutoff`` is a positive distance in angstrom. Return sorted LatticeSite
    tuples, one per motif atom, including the anchor's self image. ASE supplies
    translation offsets in the primitive lattice basis.
    """
    first, second, shifts, distances = neighbor_list(
        "ijSd", primitive.primitive_atoms, cutoff, self_interaction=True
    )
    result: list[set[LatticeSite]] = [set() for _ in range(primitive.n_atoms)]
    for anchor, site, shift, distance in zip(first, second, shifts, distances, strict=True):
        if float(distance) < cutoff:
            result[int(anchor)].add(LatticeSite(int(site), tuple(int(value) for value in shift)))
    return tuple(tuple(sorted(values)) for values in result)


def iter_candidates(
    primitive: ClusterSpace,
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
        """Return the addressed site's Cartesian row position, shape (3,), in angstrom."""
        return (positions[label.site] + np.asarray(label.translation)) @ cell

    for anchor in range(primitive.n_atoms):
        choices = neighbors[anchor]
        prefix: list[LatticeSite] = []

        def extend(
            start: int,
            *,
            choices: tuple[LatticeSite, ...] = choices,
            prefix: list[LatticeSite] = prefix,
        ) -> Iterator[tuple[LatticeSite, ...]]:
            """Yield nondecreasing neighbor sequences completing order-1 tensor slots.

            ``start`` is the first admissible index into the anchor's choices.
            ``prefix`` holds the current sequence and is restored after each
            recursive branch. All selected pairs must be within cutoff;
            repeating a neighbor is permitted. Yield tuples of LatticeSite.
            """
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


__all__ = ["iter_candidates"]
