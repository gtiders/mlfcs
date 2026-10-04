"""Construction of a primitive-only cluster space."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations
from time import perf_counter

import numpy as np

from mlfcs._arrays import require_allocation
from mlfcs.cluster_space._orbits import ClusterRegistry, _orbit_actions
from mlfcs.cluster_space.basis import _invariant_basis, component_parameterization
from mlfcs.cluster_space.candidates import _candidate_labels
from mlfcs.cluster_space.models import Cluster, Orbit, OrderBlock
from mlfcs.core.log import get_logger
from mlfcs.core.symmetry import PrimitiveSymmetry, discover_symmetry
from mlfcs.core.tensors import tensor_dimension

logger = get_logger(__name__)


def _axis_permutations(order: int) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Return all p! tensor-axis permutations and their int64 buffer after allocation checks."""
    require_allocation("axis permutations", (1, order))
    count = 1
    for factor in range(2, order + 1):
        count *= factor
        require_allocation("axis permutations", (count, order))
    values = tuple(permutations(range(order)))
    return values, np.asarray(values, dtype=np.int64)


def _tensor_frame(cell: np.ndarray, order: int) -> np.ndarray:
    """Build the (3**p, 3**p) lattice-to-Cartesian frame as cell.T Kronecker powers.

    The cell holds lattice vectors as rows. Each tensor slot receives the same
    frame; flattened components follow C order. Return a new float64 matrix.
    """
    dimension = tensor_dimension(order)
    require_allocation("tensor frame", (dimension, dimension))
    result = np.ones((1, 1), dtype=np.float64)
    for _ in range(order):
        result = np.kron(result, cell.T)
    return result


def _build_order(
    cell: np.ndarray,
    scaled_positions: np.ndarray,
    symmetry: PrimitiveSymmetry,
    *,
    order: int,
    cutoff: float,
    max_body_order: int,
) -> tuple[Orbit, ...]:
    """Enumerate one order, deduplicate its orbits and parameterize nonzero invariants.

    Input geometry uses fractional positions and row lattice vectors in angstrom.
    Candidates obey pairwise cutoff and distinct-site body order. Each orbit
    gets a saturated lattice basis, its Cartesian frame conversion and selected
    physical component parameters; zero-dimensional orbits are omitted.
    """
    candidate_labels = _candidate_labels(
        cell,
        scaled_positions,
        order=order,
        cutoff=cutoff,
        max_body_order=max_body_order,
    )
    logger.info("Candidates ready: order=%d candidates=%d", order, len(candidate_labels))
    frame = _tensor_frame(cell, order)
    permutation_values, permutation_array = _axis_permutations(order)
    covered = ClusterRegistry(order)
    generated: list[Orbit] = []

    for candidate_array in candidate_labels:
        if covered.contains(candidate_array):
            continue
        candidate = Cluster.from_labels(candidate_array)
        representative, clusters, operations, axis_permutations, stabilizers, _orbit_keys = (
            _orbit_actions(
                candidate,
                symmetry,
                permutation_values,
                permutation_array,
                return_keys=False,
            )
        )
        covered.update(np.asarray([cluster.labels for cluster in clusters], dtype=np.int64))
        exact = _invariant_basis(
            representative.sites,
            tuple((np.asarray(rotation), permutation) for rotation, permutation in stabilizers),
        )
        if exact.shape[1] == 0:
            continue
        exact_float = np.asarray(exact, dtype=np.float64)
        # Exact generators determine the invariant subspace; selected Cartesian
        # component rows determine the public, physically named coordinates.
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

    return tuple(generated)


def _construct_space(
    cell,
    scaled_positions,
    atomic_numbers,
    symprec,
    *,
    cutoffs: Mapping[int, float],
    max_body_orders: Mapping[int, int],
):
    """Discover primitive symmetry and build contiguous multi-order orbit/parameter blocks.

    Cutoff and body-order mappings must cover the same nonempty orders. Orders
    are constructed ascending; return (symmetry, blocks, orbits) without owning
    a supercell or retaining any training state.
    """
    orders = tuple(sorted(int(order) for order in cutoffs))
    if not orders or set(orders) != {int(order) for order in max_body_orders}:
        raise ValueError("cutoffs and max_body_orders must define the same nonempty orders")
    started = perf_counter()
    logger.info(
        "Cluster-space construction started: primitive_atoms=%d orders=%s symprec=%.6g",
        len(atomic_numbers),
        orders,
        symprec,
    )
    symmetry = discover_symmetry(cell, scaled_positions, atomic_numbers, symprec)
    logger.info("Primitive symmetry ready: operations=%d symbol=%s", symmetry.size, symmetry.symbol)

    all_orbits: list[Orbit] = []
    blocks: list[OrderBlock] = []
    parameter_start = 0
    for order in orders:
        order_started = perf_counter()
        logger.info(
            "Order construction started: order=%d cutoff_angstrom=%.10g max_body_order=%d",
            order,
            cutoffs[order],
            max_body_orders[order],
        )
        orbit_start = len(all_orbits)
        orbits = _build_order(
            cell,
            scaled_positions,
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
            )
        )
        logger.info(
            "Order complete: order=%d orbits=%d parameters=%d elapsed_s=%.2f",
            order,
            len(orbits),
            parameter_stop - parameter_start,
            perf_counter() - order_started,
        )
        parameter_start = parameter_stop

    logger.info(
        "Cluster-space construction complete: orders=%s orbits=%d parameters=%d elapsed_s=%.2f",
        orders,
        len(all_orbits),
        parameter_start,
        perf_counter() - started,
    )
    return symmetry, tuple(blocks), tuple(all_orbits)
