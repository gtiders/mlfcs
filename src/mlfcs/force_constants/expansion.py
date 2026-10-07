"""Primitive-lattice expansion shared by output and invariance rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.force_constants.model import ForceConstants
from mlfcs.foundation.tensors import rotate_basis, rotate_tensor


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


def image_tensor(model, orbit, representative, image):
    """Evaluate one orbit image with the model's Cartesian action convention."""
    rotation = model.cluster_space.symmetry.cartesian_rotations[orbit.operations[image]].T
    return rotate_tensor(representative, rotation, orbit.permutations[image])


def iter_lattice_tensors(model: ForceConstants, order: int):
    """Yield individual lattice labels and tensors without collecting image arrays."""
    if order not in model.coefficients:
        raise ValueError(f"force constants do not contain order {order}")
    offset = 0
    for orbit in model.cluster_space.orbits[model.cluster_space.block(order).orbits]:
        stop = offset + orbit.dimension
        representative = (orbit.component_basis @ model.coefficients[order][offset:stop]).reshape(
            (3,) * order
        )
        for image, cluster in enumerate(orbit.clusters):
            yield (
                tuple(site.site for site in cluster.sites),
                tuple(site.translation for site in cluster.sites[1:]),
                image_tensor(model, orbit, representative, image),
            )
        offset = stop
    if offset != len(model.coefficients[order]):
        raise RuntimeError("orbit traversal did not consume all force-constant coefficients")


def expand_lattice_tensors(model: ForceConstants, order: int) -> LatticeForceConstants:
    """Return all selected-order lattice-image tensors without supercell folding.

    Evaluate representative physical parameters and rotate/permutate each
    orbit image in stored order. Tensor entries have units eV/angstrom**order.
    Returns new arrays without mutating the model. Missing order raises
    ValueError; inconsistent coefficient traversal raises RuntimeError.
    """
    sites = []
    translations = []
    tensors = []
    for site, translation, tensor in iter_lattice_tensors(model, order):
        sites.append(site)
        translations.append(translation)
        tensors.append(tensor)
    return LatticeForceConstants(tuple(sites), tuple(translations), tuple(tensors))


__all__ = ["LatticeForceConstants", "expand_lattice_tensors", "rotate_basis", "rotate_tensor"]
