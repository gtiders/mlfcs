"""Assembly of primitive cluster-space records from symmetry and invariants."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations
from time import perf_counter

import numpy as np

from mlfcs.cluster_space.basis import component_parameterization, compute_invariant_basis
from mlfcs.cluster_space.candidates import enumerate_candidate_labels
from mlfcs.cluster_space.orbits import ClusterRegistry, compute_orbit_actions
from mlfcs.cluster_space.records import Cluster, Orbit, OrderBlock
from mlfcs.foundation.arrays import require_allocation
from mlfcs.foundation.tensors import tensor_dimension
from mlfcs.geometry.structure import primitive_data
from mlfcs.geometry.symmetry import PrimitiveSymmetry, discover_symmetry

_PHASE_REPORT_SECONDS = 0.25
_PROGRESS_SECONDS = 15.0


def _independent_sites(symmetry: PrimitiveSymmetry) -> int:
    """Count motif-site equivalence classes under the discovered operations.

    ``site_permutations[operation, site]`` is the image of one site, so each
    operation contributes the pairs it identifies; the classes are the orbits of
    the resulting relation. This is the number of crystallographically distinct
    sites, so a distorted input that has lost symmetry shows up as a much larger
    count than the ideal structure.
    """
    count = symmetry.site_permutations.shape[1]
    parent = list(range(count))

    def find(index: int) -> int:
        """Return the representative of one site's equivalence class."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for images in symmetry.site_permutations.tolist():
        for site, image in enumerate(images):
            root, other = find(site), find(image)
            if root != other:
                parent[other] = root
    return len({find(index) for index in range(count)})


def _axis_permutations(order: int) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Return all tensor-index permutations for the requested order."""
    require_allocation("axis permutations", (1, order))
    count = 1
    for factor in range(2, order + 1):
        count *= factor
        require_allocation("axis permutations", (count, order))
    values = tuple(permutations(range(order)))
    return values, np.asarray(values, dtype=np.int64)


def _tensor_frame(cell: np.ndarray, order: int) -> np.ndarray:
    """Return the lattice-to-Cartesian tensor frame for the requested order.

    The cell holds lattice vectors as rows. Each tensor index receives the same
    frame ``cell.T``; flattened components follow C order.
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
) -> tuple[tuple[Orbit, ...], dict]:
    """Enumerate one order and parameterize its nonzero invariants.

    Input geometry uses fractional positions and row lattice vectors in
    angstrom. Candidates obey pairwise cutoff and distinct-site body order.
    Each orbit gets a saturated lattice basis, its Cartesian frame conversion
    and selected physical component parameters; zero-dimensional orbits are
    dropped and counted.

    Returns the orbits together with a report of the enumeration: candidate,
    orbit, image, dropped and parameter counts plus phase timings, so the
    caller can print one progress line per order. This function prints its own
    throttle-limited progress line while a single order runs long.
    """
    started = perf_counter()
    candidate_labels = enumerate_candidate_labels(
        cell,
        scaled_positions,
        order=order,
        cutoff=cutoff,
        max_body_order=max_body_order,
    )
    candidates_seconds = perf_counter() - started
    frame = _tensor_frame(cell, order)
    permutation_values, permutation_array = _axis_permutations(order)
    covered = ClusterRegistry(order)
    generated: list[Orbit] = []
    images = 0
    dropped = 0
    actions_seconds = 0.0
    basis_seconds = 0.0
    parameter_seconds = 0.0
    next_report = started + _PROGRESS_SECONDS

    for processed, candidate_array in enumerate(candidate_labels, start=1):
        progress = perf_counter()
        if progress >= next_report:
            print(
                f"  FC{order}  progress: candidates {processed}/{len(candidate_labels)}, "
                f"orbits {len(generated)}, {progress - started:.0f} s elapsed"
            )
            next_report = progress + _PROGRESS_SECONDS
        if covered.contains(candidate_array):
            continue
        candidate = Cluster.from_labels(candidate_array)
        phase = perf_counter()
        representative, labels, operations, axis_permutations, stabilizers = compute_orbit_actions(
            candidate,
            symmetry,
            permutation_values,
            permutation_array,
        )
        covered.update(labels)
        images += len(labels)
        actions_seconds += perf_counter() - phase
        phase = perf_counter()
        lattice_basis = compute_invariant_basis(
            representative.sites,
            tuple((np.asarray(rotation), permutation) for rotation, permutation in stabilizers),
        )
        basis_seconds += perf_counter() - phase
        if lattice_basis.shape[1] == 0:
            dropped += 1
            continue
        phase = perf_counter()
        basis_values = np.asarray(lattice_basis, dtype=np.float64)
        # Lattice generators determine the invariant subspace; selected Cartesian
        # component rows determine the public, physically named coordinates.
        cartesian = frame @ basis_values
        component_basis, rows, condition = component_parameterization(cartesian)
        lattice_basis.setflags(write=False)
        generated.append(
            Orbit(
                representative=representative,
                labels=labels,
                lattice_basis=lattice_basis,
                component_basis=component_basis,
                observation_rows=rows,
                observation_condition=condition,
                operations=operations,
                permutations=axis_permutations,
            )
        )
        parameter_seconds += perf_counter() - phase

    report = {
        "candidates": len(candidate_labels),
        "orbits": len(generated),
        "images": images,
        "dropped": dropped,
        "parameters": sum(orbit.dimension for orbit in generated),
        "elapsed": perf_counter() - started,
        "candidates_s": candidates_seconds,
        "actions_s": actions_seconds,
        "invariants_s": basis_seconds,
        "parameters_s": parameter_seconds,
    }
    return tuple(generated), report


def construct_space(
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
    volume = abs(float(np.linalg.det(np.asarray(cell, dtype=np.float64))))
    species = len({int(number) for number in atomic_numbers})
    print(
        f"Build ClusterSpace: atoms {len(atomic_numbers)}, species {species}, "
        f"cell volume {volume:.3f} A^3, symprec {float(symprec):.1e} A"
    )
    symmetry = discover_symmetry(cell, scaled_positions, atomic_numbers, symprec)
    print(
        f"Symmetry: {symmetry.symbol}, operations {symmetry.size}, "
        f"independent sites {_independent_sites(symmetry)}"
    )
    print("Orders: " + ", ".join(f"FC{order}" for order in orders))

    all_orbits: list[Orbit] = []
    blocks: list[OrderBlock] = []
    parameter_start = 0
    for order in orders:
        orbit_start = len(all_orbits)
        orbits, report = _build_order(
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
        dropped = f" ({report['dropped']} dropped)" if report["dropped"] else ""
        print(
            f"  FC{order}  cutoff {float(cutoffs[order]):g} A  "
            f"body {int(max_body_orders[order])}  "
            f"candidates {report['candidates']}  "
            f"orbits {report['orbits']}{dropped}  "
            f"images {report['images']}  "
            f"parameters {parameter_stop - parameter_start}  "
            f"{report['elapsed']:.2f} s"
        )
        if report["elapsed"] >= _PHASE_REPORT_SECONDS:
            print(
                f"        phases: candidates {report['candidates_s']:.2f} s, "
                f"orbit actions {report['actions_s']:.2f} s, "
                f"invariants {report['invariants_s']:.2f} s, "
                f"parameters {report['parameters_s']:.2f} s"
            )
        if not report["orbits"]:
            print(
                f"  warning: FC{order} has no symmetry-allowed orbit at cutoff "
                f"{float(cutoffs[order]):g} A with body order {int(max_body_orders[order])}"
            )
        parameter_start = parameter_stop

    print(
        f"Total: orbits {len(all_orbits)}, parameters {parameter_start}, "
        f"{perf_counter() - started:.2f} s"
    )
    return symmetry, tuple(blocks), tuple(all_orbits)


def assemble_parts(primitive_atoms, *, cutoffs, max_body_orders, symprec):
    """Derive validated record parts for a primitive cluster space.

    ``cutoffs`` and ``max_body_orders`` are already normalized integer-keyed
    mappings with identical key sets. Acoustic preparation is not performed
    here; the record constructor requests it separately.
    """
    primitive = primitive_data(primitive_atoms, symprec)
    symmetry, blocks, orbits = construct_space(
        primitive.cell,
        primitive.scaled_positions,
        primitive.atomic_numbers,
        primitive.symprec,
        cutoffs=cutoffs,
        max_body_orders=max_body_orders,
    )
    return {
        "cell": primitive.cell,
        "scaled_positions": primitive.scaled_positions,
        "atomic_numbers": primitive.atomic_numbers,
        "symprec": primitive.symprec,
        "symmetry": symmetry,
        "blocks": blocks,
        "orbits": orbits,
        "masses": primitive.masses,
        "acoustic_coordinates": None,
    }
