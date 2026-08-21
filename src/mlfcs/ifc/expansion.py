"""Shared host-side expansion of irreducible orbit parameters."""

from __future__ import annotations

import numpy as np

from mlfcs.core.real_space import PrimitiveInteractionSpace
from mlfcs.ifc.model import SparseOrderForceConstants


def expand_primitive_parameters(
    interaction_space: PrimitiveInteractionSpace,
    parameters: np.ndarray,
) -> SparseOrderForceConstants:
    """Expand primitive-orbit parameters into canonical exact-R IFC rows."""
    values = np.asarray(parameters, dtype=float).reshape(-1)
    expected = sum(orbit.dimension for orbit in interaction_space.orbits)
    if len(values) != expected:
        raise ValueError(f"expected {expected} orbit parameters, got {len(values)}")
    sites = []
    translations = []
    tensors = []
    offset = 0
    shape = (3,) * interaction_space.order
    for orbit in interaction_space.orbits:
        pivot_values = values[offset : offset + orbit.dimension]
        offset += orbit.dimension
        representative = orbit.basis @ np.linalg.solve(orbit.basis[orbit.pivots], pivot_values)
        for image in orbit.images:
            key = image.key
            sites.append(key.sites)
            translations.append(key.translations)
            tensors.append(image.action.apply_flat(representative).reshape(shape))
    return SparseOrderForceConstants(
        interaction_space.order,
        np.asarray(sites, dtype=np.int32),
        np.asarray(translations, dtype=np.int32),
        np.asarray(tensors, dtype=float),
    )


__all__ = ["expand_primitive_parameters"]
