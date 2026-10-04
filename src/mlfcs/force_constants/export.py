"""Shared expansion for external force-constant formats."""

from __future__ import annotations

import numpy as np

from mlfcs.core import LatticeSite
from mlfcs.force_constants.lattice import expand
from mlfcs.force_constants.model import ForceConstants
from mlfcs.mapping import ClusterMap

DEFAULT_THRESHOLD = 1e-8


def threshold_value(value: object) -> float:
    """Validate and return one external-output cleanup threshold."""
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("threshold must be a finite non-negative number")
    return result


def clean(values: object, threshold: float) -> np.ndarray:
    """Return finite float64 values with components below ``threshold`` zeroed."""
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError("expanded force constants contain NaN or infinite values")
    result[np.abs(result) < threshold_value(threshold)] = 0.0
    return result


def validate(model: ForceConstants, cluster_map: ClusterMap, order: int) -> None:
    """Validate the model, cluster map and requested tensor order."""
    if not isinstance(cluster_map, ClusterMap):
        raise TypeError("mapping must be a ClusterMap")
    if model.cluster_space.fingerprint != cluster_map.cluster_space.fingerprint:
        raise ValueError("force constants and cluster map use different cluster spaces")
    if order not in model.coefficients:
        raise ValueError(f"force constants do not contain order {order}")


def compact(
    model: ForceConstants,
    cluster_map: ClusterMap,
    order: int,
    *,
    threshold: float,
) -> np.ndarray:
    """Return primitive-first Cartesian tensors folded into the target supercell.

    Output shape is (N_primitive, N_super, ..., N_super, 3, ..., 3), with order
    atom and Cartesian axes. Image aliases accumulate before components below
    threshold are zeroed. Output is a new writable float64 array in physical
    units; the primitive model and mapping remain unchanged.
    """
    validate(model, cluster_map, order)
    expanded = expand(model, order)
    supercell = cluster_map
    shape = (model.cluster_space.n_atoms,) + (len(supercell.atomic_numbers),) * (order - 1)
    result = np.zeros(shape + (3,) * order, dtype=np.float64)
    for sites, translations, tensor in zip(
        expanded.sites, expanded.translations, expanded.tensors, strict=True
    ):
        atoms = tuple(
            supercell.atom_index(LatticeSite(site, translation))
            for site, translation in zip(sites[1:], translations, strict=True)
        )
        result[(sites[0], *atoms)] += tensor
    return clean(result, threshold)


def translated_atoms(cluster_map: ClusterMap, first: int) -> np.ndarray:
    """Map the explicit supercell order into coordinates relative to ``first``."""
    supercell = cluster_map
    origin = supercell.lattice_translations[first]
    result = np.empty(len(supercell.atomic_numbers), dtype=np.int64)
    for atom, (site, translation) in enumerate(
        zip(supercell.primitive_site_indices, supercell.lattice_translations, strict=True)
    ):
        relative = tuple(
            int(value) - int(zero) for value, zero in zip(translation, origin, strict=True)
        )
        result[atom] = supercell.atom_index(LatticeSite(int(site), relative))
    return result


def full_fc2(compact_values: np.ndarray, cluster_map: ClusterMap) -> np.ndarray:
    """Expand primitive-first FC2 into the explicit full-supercell order."""
    supercell = cluster_map
    size = len(supercell.atomic_numbers)
    result = np.empty((size, size, 3, 3), dtype=np.float64)
    for first in range(size):
        tails = translated_atoms(cluster_map, first)
        result[first] = compact_values[int(supercell.primitive_site_indices[first]), tails]
    return result


def primitive_to_supercell(cluster_map: ClusterMap) -> np.ndarray:
    """Return the first explicit supercell atom for every primitive site."""
    sites = cluster_map.primitive_site_indices
    return np.asarray(
        [np.flatnonzero(sites == site)[0] for site in range(cluster_map.cluster_space.n_atoms)]
    )


__all__ = [
    "DEFAULT_THRESHOLD",
    "clean",
    "compact",
    "full_fc2",
    "primitive_to_supercell",
    "threshold_value",
    "translated_atoms",
    "validate",
]
