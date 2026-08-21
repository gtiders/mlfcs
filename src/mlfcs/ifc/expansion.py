"""Shared host-side expansion of irreducible orbit parameters."""

from __future__ import annotations

import numpy as np

from mlfcs.core.geometry import PeriodicIndex
from mlfcs.core.orbits import OrbitSpace
from mlfcs.core.real_space import PrimitiveInteractionSpace
from mlfcs.ifc.model import SparseOrderForceConstants


def expand_orbit_parameters(
    orbit_space: OrbitSpace,
    parameters: np.ndarray,
    *,
    n_primitive: int,
    n_supercell: int,
    index: PeriodicIndex | None = None,
) -> SparseOrderForceConstants:
    """Expand one order's pivot-value coordinates into sparse IFC tensors.

    Orbit bases are normalized to deterministic pivot-value coordinates, so
    this common operation applies equally to finite-difference derivatives and
    Taylor parameters recovered from a fit.
    """
    values = np.asarray(parameters, dtype=float).reshape(-1)
    expected = sum(orbit.dimension for orbit in orbit_space.orbits)
    if len(values) != expected:
        raise ValueError(f"expected {expected} orbit parameters, got {len(values)}")

    clusters: list[tuple[int, ...]] = []
    tensors: list[np.ndarray] = []
    offset = 0
    shape = (3,) * orbit_space.order
    for orbit in orbit_space.orbits:
        pivot_values = values[offset : offset + orbit.dimension]
        offset += orbit.dimension
        representative = orbit.basis @ np.linalg.solve(orbit.basis[orbit.pivots], pivot_values)
        for image in orbit.images:
            clusters.append(image.cluster)
            tensors.append(image.action.apply_flat(representative).reshape(shape))

    cluster_array = np.asarray(clusters, dtype=np.int32).reshape((-1, orbit_space.order))
    sites = translations = None
    if index is not None:
        sites = index.primitive[cluster_array]
        raw = index.translations[cluster_array[:, 1:]] - index.translations[cluster_array[:, :1]]
        translations = np.asarray(
            [[index.canonical_translation(vector) for vector in row] for row in raw], dtype=np.int32
        )
    return SparseOrderForceConstants(
        orbit_space.order,
        n_primitive,
        n_supercell,
        cluster_array,
        np.asarray(tensors, dtype=float).reshape((-1,) + shape),
        sites,
        translations,
    )


def expand_primitive_parameters(
    interaction_space: PrimitiveInteractionSpace,
    parameters: np.ndarray,
    *,
    index: PeriodicIndex,
) -> SparseOrderForceConstants:
    """Expand primitive-orbit parameters with exact lattice translations.

    ``clusters`` remains a finite calculation view during the staged solver
    migration.  ``sites`` and ``translations`` already carry
    the canonical exact-R physical identity and are independent of that view.
    """
    values = np.asarray(parameters, dtype=float).reshape(-1)
    expected = sum(orbit.dimension for orbit in interaction_space.orbits)
    if len(values) != expected:
        raise ValueError(f"expected {expected} orbit parameters, got {len(values)}")
    clusters = []
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
            cluster = [index.representative(key.sites[0])]
            cluster.extend(
                index.atom(site, translation)
                for site, translation in zip(key.sites[1:], key.translations, strict=True)
            )
            clusters.append(cluster)
            sites.append(key.sites)
            translations.append(key.translations)
            tensors.append(image.action.apply_flat(representative).reshape(shape))
    return SparseOrderForceConstants(
        interaction_space.order,
        index.n_primitive,
        len(index.primitive),
        np.asarray(clusters, dtype=np.int32),
        np.asarray(tensors, dtype=float),
        np.asarray(sites, dtype=np.int32),
        np.asarray(translations, dtype=np.int32),
    )


__all__ = ["expand_orbit_parameters", "expand_primitive_parameters"]
