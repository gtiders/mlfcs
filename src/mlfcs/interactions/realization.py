"""Realization and identifiability of primitive interactions in one reference."""

from __future__ import annotations

import numpy as np

from mlfcs.exceptions import InteractionAliasingError
from mlfcs.interactions.algebra.exact import certified_rank
from mlfcs.interactions.keys import InteractionKey
from mlfcs.interactions.models import (
    PrimitiveInteractionSpace,
    RealizedInteractionOrbit,
    RealizedInteractionSpace,
    RealizedOrbitImage,
)


def validate_realization_identifiability(
    space: PrimitiveInteractionSpace,
    index,
    *,
    realized: RealizedInteractionSpace | None = None,
) -> None:
    """Reject a finite reference that cannot identify primitive parameters.

    Every coefficient is an exact integer: the orbit basis is its lattice (scaled) form,
    the tensor actions are the integer lattice rotations, and each column-graph component
    is ranked with the modular certificate.  No coefficient filter and no rank tolerance
    is involved, so a genuinely non-zero coefficient can never be discarded.
    """
    if realized is not None and len(realized.orbits) != len(space.orbits):
        raise ValueError("realized and primitive orbit spaces are inconsistent")
    parameter_offsets = np.cumsum([0, *(orbit.dimension for orbit in space.orbits)], dtype=np.int64)
    n_parameters = int(parameter_offsets[-1])
    rows: dict[tuple[tuple[int, ...], int], dict[int, int]] = {}
    for orbit_index, orbit in enumerate(space.orbits):
        offset = int(parameter_offsets[orbit_index])
        for image_index, image in enumerate(orbit.images):
            cluster = _realized_cluster(image, orbit_index, image_index, index, realized)
            columns = image.action.apply_scaled_columns(orbit.exact_lattice_basis)
            for component, values in enumerate(columns):
                row = rows.setdefault((cluster, component), {})
                for local, value in enumerate(values):
                    if value:
                        column = offset + local
                        row[column] = row.get(column, 0) + int(value)

    parent = np.arange(n_parameters, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    row_values: list[dict[int, int]] = []
    for values in rows.values():
        non_zero = {column: numerator for column, numerator in values.items() if numerator}
        if not non_zero:
            continue
        row_values.append(non_zero)
        columns = tuple(non_zero)
        for column in columns[1:]:
            union(columns[0], column)
    components: dict[int, list[int]] = {}
    for column in range(n_parameters):
        components.setdefault(find(column), []).append(column)

    component_rows: dict[int, list[dict[int, int]]] = {}
    for values in row_values:
        component_rows.setdefault(find(next(iter(values))), []).append(values)
    for root, columns in components.items():
        local_rows = component_rows.get(root, [])
        integer_rows = [[values.get(column, 0) for column in columns] for values in local_rows]
        rank = certified_rank([row for row in integer_rows if any(row)])
        if rank != len(columns):
            _raise_aliasing(space, parameter_offsets, columns, rank)


def _realized_cluster(image, orbit_index: int, image_index: int, index, realized):
    if realized is None:
        return _realize_key(image.key, index)
    return realized.orbits[orbit_index].images[image_index].cluster


def _raise_aliasing(
    space: PrimitiveInteractionSpace,
    parameter_offsets: np.ndarray,
    columns: tuple[int, ...] | list[int],
    rank: int,
) -> None:
    affected = [
        orbit.representative
        for orbit_index, orbit in enumerate(space.orbits)
        if any(
            int(parameter_offsets[orbit_index]) <= column < int(parameter_offsets[orbit_index + 1])
            for column in columns
        )
    ]
    raise InteractionAliasingError(
        f"source reference identifies only {rank} of {len(columns)} independent "
        f"FC{space.order} parameters in a folded realization component; "
        f"conflicting primitive interactions include {affected[:4]}. "
        "Use a larger single reference supercell or a shorter cutoff."
    )


def _realize_orbit_space(space: PrimitiveInteractionSpace, index) -> RealizedInteractionSpace:
    """Realize an exact primitive orbit space in one finite reference frame."""
    realized: list[RealizedInteractionOrbit] = []
    for orbit in space.orbits:
        representative = _realize_key(orbit.representative, index)
        images = []
        for image in orbit.images:
            cluster = _realize_key(image.key, index)
            # Duplicate concrete clusters are intentional here: a small
            # reference can fold several exact-R images onto the same atoms.
            # The design kernel scatters every image contribution and thereby
            # forms the correct periodized sum.  Identifiability is a property
            # of the complete constrained design, not of this local mapping.
            images.append(RealizedOrbitImage(cluster, image.action))
        realized.append(
            RealizedInteractionOrbit(
                representative,
                orbit.cartesian_basis,
                orbit.observation_rows,
                tuple(images),
            )
        )
    return RealizedInteractionSpace(
        space.order, tuple(realized), space.cutoff, space.max_body_order
    )


def realize_interaction_space(
    space: PrimitiveInteractionSpace,
    index,
    *,
    validate_identifiability: bool = True,
) -> RealizedInteractionSpace:
    """Realize primitive exact-R interactions in one reference cell."""
    result = _realize_orbit_space(space, index)
    if validate_identifiability:
        validate_realization_identifiability(space, index, realized=result)
    return result


def _realize_key(key: InteractionKey, index) -> tuple[int, ...]:
    atoms = [index.representative(key.sites[0])]
    for site, translation in zip(key.sites[1:], key.translations, strict=True):
        atoms.append(index.atom(int(site), translation))
    return tuple(atoms)


__all__ = ["realize_interaction_space", "validate_realization_identifiability"]
