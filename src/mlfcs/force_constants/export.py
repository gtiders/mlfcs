"""Shared expansion for external force-constant formats."""

from __future__ import annotations

import numpy as np

from mlfcs.core import LatticeSite
from mlfcs.force_constants.lattice import expand
from mlfcs.force_constants.model import ForceConstants
from mlfcs.supercell import ClusterMap

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


def validate(model: ForceConstants, mapping: ClusterMap, order: int) -> None:
    """Validate the model, cluster map and requested tensor order."""
    if not isinstance(mapping, ClusterMap):
        raise TypeError("mapping must be a ClusterMap")
    if model.space.fingerprint != mapping.space.fingerprint:
        raise ValueError("force constants and cluster map use different cluster spaces")
    if order not in model.coefficients:
        raise ValueError(f"force constants do not contain order {order}")


def compact(
    model: ForceConstants,
    mapping: ClusterMap,
    order: int,
    *,
    threshold: float,
) -> np.ndarray:
    """Return primitive-first force constants folded into ``mapping``."""
    validate(model, mapping, order)
    expanded = expand(model, order)
    supercell = mapping.supercell
    shape = (model.space.primitive.size,) + (len(supercell.numbers),) * (order - 1)
    result = np.zeros(shape + (3,) * order, dtype=np.float64)
    for sites, translations, tensor in zip(
        expanded.sites, expanded.translations, expanded.tensors, strict=True
    ):
        atoms = tuple(
            supercell.atom(LatticeSite(site, translation))
            for site, translation in zip(sites[1:], translations, strict=True)
        )
        result[(sites[0], *atoms)] += tensor
    return clean(result, threshold)


def translated_atoms(mapping: ClusterMap, first: int) -> np.ndarray:
    """Map the explicit supercell order into coordinates relative to ``first``."""
    supercell = mapping.supercell
    origin = supercell.translations[first]
    result = np.empty(len(supercell.numbers), dtype=np.int64)
    for atom, (site, translation) in enumerate(
        zip(supercell.sites, supercell.translations, strict=True)
    ):
        relative = tuple(int(value - zero) for value, zero in zip(translation, origin, strict=True))
        result[atom] = supercell.atom(LatticeSite(int(site), relative))
    return result


def full_fc2(compact_values: np.ndarray, mapping: ClusterMap) -> np.ndarray:
    """Expand primitive-first FC2 into the explicit full-supercell order."""
    supercell = mapping.supercell
    size = len(supercell.numbers)
    result = np.empty((size, size, 3, 3), dtype=np.float64)
    for first in range(size):
        tails = translated_atoms(mapping, first)
        result[first] = compact_values[int(supercell.sites[first]), tails]
    return result


def primitive_to_supercell(mapping: ClusterMap) -> np.ndarray:
    """Return the first explicit supercell atom for every primitive site."""
    sites = mapping.supercell.sites
    return np.asarray(
        [np.flatnonzero(sites == site)[0] for site in range(mapping.space.primitive.size)]
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
