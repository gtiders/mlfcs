"""Primitive-lattice expansion shared by output and invariance rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.force_constants.model import ForceConstants
from mlfcs.tensors import rotate_basis, rotate_tensor


@dataclass(frozen=True, slots=True)
class LatticeForceConstants:
    """Cartesian tensors expanded onto anchored primitive-lattice labels.

    For order p, sites contains p primitive indices per image, translations
    contains only the last p-1 integer shifts (the first is zero), and tensors
    contains newly evaluated arrays of shape (3,) repeated p times. These arrays
    are not marked readonly; dataclass freezing only freezes field assignment.
    """

    sites: tuple[tuple[int, ...], ...]
    translations: tuple[tuple[tuple[int, int, int], ...], ...]
    tensors: tuple[np.ndarray, ...]


def expand_lattice_tensors(model: ForceConstants, order: int) -> LatticeForceConstants:
    """Return all selected-order lattice-image tensors without supercell folding.

    Evaluate representative physical parameters and rotate/permutate each
    orbit image in stored order. Tensor entries have units eV/angstrom**order.
    Returns new arrays without mutating the model. Missing order raises
    ValueError; inconsistent coefficient traversal raises RuntimeError.
    """
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


__all__ = ["LatticeForceConstants", "expand_lattice_tensors", "rotate_basis", "rotate_tensor"]
