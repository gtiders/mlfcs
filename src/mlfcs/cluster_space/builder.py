"""Construction of a primitive-only cluster space."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations

import numpy as np

from mlfcs.cluster_space._orbit_kernel import transform_cluster
from mlfcs.cluster_space.candidates import iter_candidates
from mlfcs.cluster_space.invariants import invariant_basis
from mlfcs.cluster_space.models import Cluster, ClusterSpace, IntBounds, Orbit, OrderBlock
from mlfcs.cluster_space.observation import component_parameterization
from mlfcs.core import PrimitiveCell, PrimitiveSymmetry
from mlfcs.core.log import get_logger

logger = get_logger(__name__)


def _axis_permutations(order: int) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    values = tuple(permutations(range(order)))
    return values, np.asarray(values, dtype=np.int64)


def _integer_boundary(
    labels: tuple[tuple[int, int, int, int], ...], symmetry: PrimitiveSymmetry
) -> tuple[np.ndarray, int, int, int, int]:
    """Prove that one cluster action fits int64, without a numerical tolerance."""
    maximum_translation = max((abs(value) for row in labels for value in row[1:]), default=0)
    maximum_rotation = max((abs(int(value)) for value in symmetry.rotations.flat), default=0)
    maximum_shift = max((abs(int(value)) for value in symmetry.site_shifts.flat), default=0)
    unanchored = 3 * maximum_translation * maximum_rotation + maximum_shift
    bound = 2 * unanchored
    limit = np.iinfo(np.int64).max
    if bound > limit:
        raise OverflowError(
            "the exact primitive cluster action cannot enter the int64 compiled kernel: "
            f"its proven bound is {bound}, above {limit}"
        )
    array = np.asarray(labels, dtype=object)
    if any(abs(int(value)) > limit for value in array.flat):
        raise OverflowError("primitive cluster labels do not fit the int64 compiled kernel")
    return (
        np.asarray(array, dtype=np.int64),
        maximum_translation,
        maximum_rotation,
        maximum_shift,
        bound,
    )


def _cluster_key(values: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in values.reshape(-1))


def _cluster_from_key(values: tuple[int, ...], order: int) -> Cluster:
    return Cluster.from_labels(np.asarray(values, dtype=object).reshape(order, 4))


def _orbit_actions(
    cluster: Cluster,
    symmetry: PrimitiveSymmetry,
    permutation_values: tuple[tuple[int, ...], ...],
    permutation_array: np.ndarray,
) -> tuple[
    Cluster,
    tuple[Cluster, ...],
    np.ndarray,
    np.ndarray,
    tuple[tuple[np.ndarray, tuple[int, ...]], ...],
    tuple[tuple[int, ...], ...],
]:
    labels, *_ = _integer_boundary(cluster.labels, symmetry)
    transformed = transform_cluster(
        labels,
        np.asarray(symmetry.rotations, dtype=np.int64),
        np.asarray(symmetry.site_permutations, dtype=np.int64),
        np.asarray(symmetry.site_shifts, dtype=np.int64),
        permutation_array,
    )
    representative = cluster
    representative_key = _cluster_key(labels)
    images: dict[tuple[int, ...], tuple[Cluster, int, tuple[int, ...]]] = {}
    stabilizers: list[tuple[np.ndarray, tuple[int, ...]]] = []
    permutation_count = len(permutation_values)
    for index, row in enumerate(transformed):
        operation, permutation_index = divmod(index, permutation_count)
        key = _cluster_key(row)
        permutation = permutation_values[permutation_index]
        images.setdefault(key, (_cluster_from_key(key, cluster.order), operation, permutation))
        if key == representative_key:
            stabilizers.append((symmetry.rotations[operation], permutation))
    if representative_key not in images or not stabilizers:
        raise RuntimeError("the primitive symmetry group does not contain the identity action")
    ordered = tuple(images.values())
    clusters = tuple(value[0] for value in ordered)
    operations = np.asarray([value[1] for value in ordered], dtype=np.int32)
    permutations = np.asarray([value[2] for value in ordered], dtype=np.int16)
    return (
        representative,
        clusters,
        operations,
        permutations,
        tuple(stabilizers),
        tuple(images),
    )


def _tensor_frame(cell: np.ndarray, order: int) -> np.ndarray:
    result = np.ones((1, 1), dtype=np.float64)
    for _ in range(order):
        result = np.kron(result, cell.T)
    return result


def _build_order(
    primitive: PrimitiveCell,
    symmetry: PrimitiveSymmetry,
    *,
    order: int,
    cutoff: float,
    max_body_order: int,
) -> tuple[tuple[Orbit, ...], IntBounds]:
    """Build one tensor-order block for :func:`build_cluster_space`."""
    permutation_values, permutation_array = _axis_permutations(order)
    covered: set[tuple[int, ...]] = set()
    generated: list[Orbit] = []
    maximum_translation = 0
    maximum_rotation = max(abs(int(value)) for value in symmetry.rotations.flat)
    maximum_shift = max(abs(int(value)) for value in symmetry.site_shifts.flat)
    maximum_bound = 0
    maximum_kernel = 0
    frame = _tensor_frame(primitive.cell, order)

    for candidate in iter_candidates(
        primitive,
        order=order,
        cutoff=cutoff,
        max_body_order=max_body_order,
    ):
        candidate_key = tuple(value for row in candidate.labels for value in row)
        if candidate_key in covered:
            continue
        labels, translation, _rotation, _shift, bound = _integer_boundary(
            candidate.labels, symmetry
        )
        del labels
        maximum_translation = max(maximum_translation, translation)
        maximum_bound = max(maximum_bound, bound)
        representative, clusters, operations, axis_permutations, stabilizers, orbit_keys = (
            _orbit_actions(candidate, symmetry, permutation_values, permutation_array)
        )
        covered.update(orbit_keys)
        exact = invariant_basis(
            representative.sites,
            tuple((np.asarray(rotation), permutation) for rotation, permutation in stabilizers),
        )
        if exact.shape[1] == 0:
            continue
        maximum_kernel = max(maximum_kernel, max(abs(int(value)) for value in exact.flat))
        exact_float = np.asarray(exact, dtype=np.float64)
        cartesian = frame @ exact_float
        component_basis, rows, condition = component_parameterization(cartesian)
        exact.setflags(write=False)
        generated.append(
            Orbit(
                representative=representative,
                exact_lattice_basis=exact,
                component_basis=component_basis,
                observation_rows=rows,
                observation_condition=condition,
                clusters=clusters,
                operations=operations,
                permutations=axis_permutations,
            )
        )

    bounds = IntBounds(
        translation=maximum_translation,
        rotation=maximum_rotation,
        shift=maximum_shift,
        labels=maximum_bound,
        headroom=np.iinfo(np.int64).max.bit_length() - max(1, maximum_bound).bit_length(),
        tensor=(3 * maximum_rotation) ** order + 1,
        kernel=maximum_kernel,
    )
    return tuple(generated), bounds


def build_cluster_space(
    primitive: PrimitiveCell,
    *,
    cutoffs: Mapping[int, float],
    max_body_orders: Mapping[int, int],
    symmetry: PrimitiveSymmetry | None = None,
) -> ClusterSpace:
    """Build a multi-order primitive cluster space."""
    orders = tuple(sorted(int(order) for order in cutoffs))
    if not orders or set(orders) != {int(order) for order in max_body_orders}:
        raise ValueError("cutoffs and max_body_orders must define the same nonempty orders")
    symmetry = PrimitiveSymmetry.from_primitive(primitive) if symmetry is None else symmetry
    if symmetry.symprec != primitive.symprec:
        raise ValueError("primitive and symmetry must use the same declared symprec")
    if symmetry.site_permutations.shape[1] != primitive.size:
        raise ValueError("primitive symmetry acts on a different motif size")

    all_orbits: list[Orbit] = []
    blocks: list[OrderBlock] = []
    parameter_start = 0
    for order in orders:
        logger.info(
            "Building FC%d cluster space: cutoff %.10g Å, maximum body order %d",
            order,
            cutoffs[order],
            max_body_orders[order],
        )
        orbit_start = len(all_orbits)
        orbits, bounds = _build_order(
            primitive,
            symmetry,
            order=order,
            cutoff=float(cutoffs[order]),
            max_body_order=int(max_body_orders[order]),
        )
        all_orbits.extend(orbits)
        parameter_stop = parameter_start + sum(orbit.dimension for orbit in orbits)
        blocks.append(
            OrderBlock(
                order=order,
                cutoff=float(cutoffs[order]),
                max_body_order=int(max_body_orders[order]),
                orbits=slice(orbit_start, len(all_orbits)),
                parameters=slice(parameter_start, parameter_stop),
                bounds=bounds,
            )
        )
        logger.info(
            "FC%d: %d orbits, %d parameters",
            order,
            len(orbits),
            parameter_stop - parameter_start,
        )
        parameter_start = parameter_stop

    return ClusterSpace(
        primitive=primitive,
        symmetry=symmetry,
        blocks=tuple(blocks),
        orbits=tuple(all_orbits),
    )


__all__ = ["build_cluster_space"]
