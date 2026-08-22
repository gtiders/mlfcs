"""Realization and identifiability of primitive interactions in one reference."""

from __future__ import annotations

import numpy as np

from mlfcs.interactions.keys import InteractionKey
from mlfcs.interactions.orbits import (
    ClusterOrbit,
    OrbitImage,
    OrbitSpace,
    PrimitiveInteractionSpace,
)


class InteractionAliasingError(ValueError):
    """A finite reference folds distinct primitive interactions together."""


def validate_realization_identifiability(
    space: PrimitiveInteractionSpace,
    index,
    *,
    tolerance: float = 1e-10,
    realized: OrbitSpace | None = None,
) -> None:
    """Reject a finite reference that cannot identify primitive parameters.

    The realization matrix is assembled in concrete IFC-component space.
    Its column graph normally separates into small independent components, so
    exact rank tests do not require a dense global matrix.
    """
    parameter_offsets = np.cumsum([0, *(orbit.dimension for orbit in space.orbits)], dtype=np.int64)
    rows: dict[tuple[tuple[int, ...], int], dict[int, float]] = {}
    if realized is not None and len(realized.orbits) != len(space.orbits):
        raise ValueError("realized and primitive orbit spaces are inconsistent")
    for orbit_index, orbit in enumerate(space.orbits):
        offset = int(parameter_offsets[orbit_index])
        realized_images = None if realized is None else realized.orbits[orbit_index].images
        for image_index, image in enumerate(orbit.images):
            cluster = (
                _realize_key(image.key, index)
                if realized_images is None
                else realized_images[image_index].cluster
            )
            columns = image.action.apply_columns(orbit.basis)
            for component, values in enumerate(columns):
                row = rows.setdefault((cluster, component), {})
                for local, value in enumerate(values):
                    if abs(value) > tolerance:
                        column = offset + local
                        row[column] = row.get(column, 0.0) + float(value)

    n_parameters = int(parameter_offsets[-1])
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

    row_values = []
    for values in rows.values():
        values = {column: value for column, value in values.items() if abs(value) > tolerance}
        if not values:
            continue
        row_values.append(values)
        columns = tuple(values)
        for column in columns[1:]:
            union(columns[0], column)
    components: dict[int, list[int]] = {}
    for column in range(n_parameters):
        components.setdefault(find(column), []).append(column)

    component_rows: dict[int, list[dict[int, float]]] = {}
    for values in row_values:
        root = find(next(iter(values)))
        component_rows.setdefault(root, []).append(values)
    for root, columns in components.items():
        local_rows = component_rows.get(root, [])
        if len(columns) == 1:
            column = columns[0]
            if any(abs(values.get(column, 0.0)) > tolerance for values in local_rows):
                continue
            rank = 0
        else:
            matrix = np.asarray(
                [[values.get(column, 0.0) for column in columns] for values in local_rows],
                dtype=float,
            )
            rank = int(np.linalg.matrix_rank(matrix, tol=tolerance))
        if rank != len(columns):
            affected = [
                orbit.representative
                for orbit_index, orbit in enumerate(space.orbits)
                if any(
                    int(parameter_offsets[orbit_index])
                    <= column
                    < int(parameter_offsets[orbit_index + 1])
                    for column in columns
                )
            ]
            raise InteractionAliasingError(
                f"source reference identifies only {rank} of {len(columns)} independent "
                f"FC{space.order} parameters in a folded realization component; "
                f"conflicting primitive interactions include {affected[:4]}. "
                "Use a larger single reference supercell or a shorter cutoff."
            )


def realize_orbit_space(space: PrimitiveInteractionSpace, index) -> OrbitSpace:
    """Realize an exact primitive orbit space in one finite reference frame."""
    realized: list[ClusterOrbit] = []
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
            images.append(OrbitImage(cluster, image.action))
        realized.append(
            ClusterOrbit(
                representative,
                orbit.basis,
                orbit.pivots,
                tuple(images),
            )
        )
    return OrbitSpace(space.order, tuple(realized), space.cutoff, space.max_body_order)


def _realize_key(key: InteractionKey, index) -> tuple[int, ...]:
    atoms = [index.representative(key.sites[0])]
    atoms.extend(
        index.atom(site, translation)
        for site, translation in zip(key.sites[1:], key.translations, strict=True)
    )
    return tuple(atoms)
