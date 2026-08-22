"""Shared host-side expansion of irreducible orbit parameters."""

from __future__ import annotations

import numpy as np

from mlfcs.force_constants.data import SparseOrderForceConstants
from mlfcs.interactions.orbits import PrimitiveInteractionSpace


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


def expand_fitted_orders(parameters, calculations):
    """Expand a packed multi-order parameter vector into canonical sparse IFCs."""
    result = {}
    offset = 0
    for calculation in calculations:
        count = sum(orbit.dimension for orbit in calculation.orbit_space.orbits)
        primitive_space = getattr(calculation, "primitive_orbit_space", None)
        if primitive_space is None:
            primitive_space = calculation.interaction_space.primitive_orbit_space
        result[calculation.config.order] = expand_primitive_parameters(
            primitive_space, parameters[offset : offset + count]
        )
        offset += count
    return result


__all__ = ["expand_fitted_orders", "expand_primitive_parameters"]
