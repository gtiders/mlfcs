"""Primitive-lattice expansion shared by output and invariance rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.core.tensors import rotate_basis, rotate_tensor
from mlfcs.force_constants.model import ForceConstants


@dataclass(frozen=True, slots=True)
class LatticeForceConstants:
    """One order expanded onto exact primitive-lattice cluster labels."""

    sites: tuple[tuple[int, ...], ...]
    translations: tuple[tuple[tuple[int, int, int], ...], ...]
    tensors: tuple[np.ndarray, ...]


def expand(model: ForceConstants, order: int) -> LatticeForceConstants:
    """Expand orbit coefficients without choosing or folding into a supercell."""
    if order not in model.coefficients:
        raise ValueError(f"force constants do not contain order {order}")
    block = model.cluster_space.block(order)
    coefficients = model.coefficients[order]
    offset = 0
    sites = []
    translations = []
    tensors = []
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        orbit = model.cluster_space.orbits[orbit_index]
        stop = offset + orbit.dimension
        representative = (orbit.component_basis @ coefficients[offset:stop]).reshape((3,) * order)
        offset = stop
        for image, cluster in enumerate(orbit.clusters):
            rotation = model.cluster_space.symmetry.cartesian_rotations[orbit.operations[image]].T
            tensors.append(rotate_tensor(representative, rotation, orbit.permutations[image]))
            sites.append(tuple(site.site for site in cluster.sites))
            translations.append(tuple(site.translation for site in cluster.sites[1:]))
    if offset != len(coefficients):
        raise RuntimeError(
            f"expanded {offset} order-{order} coefficients, expected {len(coefficients)}"
        )
    return LatticeForceConstants(tuple(sites), tuple(translations), tuple(tensors))


__all__ = ["LatticeForceConstants", "expand", "rotate_basis", "rotate_tensor"]
