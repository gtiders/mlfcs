"""Exact structural rank of folded cluster images."""

from dataclasses import dataclass

import numpy as np
from numba import njit

from mlfcs._arrays import require_allocation, require_bound
from mlfcs.algebra.exact import exact_rank
from mlfcs.core.tensors import _tensor_action, tensor_action_bound
from mlfcs.errors import AliasingError


@dataclass(frozen=True, slots=True)
class RankInfo:
    """Exact structural rank of a cluster map."""

    parameters: int
    rank: int
    aliases: int

    @property
    def nullity(self) -> int:
        return self.parameters - self.rank

    @property
    def full(self) -> bool:
        return self.rank == self.parameters

    def require_full(self) -> None:
        """Raise when the supercell leaves primitive parameters aliased."""
        if not self.full:
            raise AliasingError(
                f"supercell realization has rank {self.rank} for {self.parameters} "
                f"parameters (nullity {self.nullity})"
            )


def folded_rank(cluster_space, image_atom_indices, order=None):
    """Return exact structural rank, independently of training displacements."""
    if order is None:
        values = [
            folded_rank(cluster_space, image_atom_indices, value) for value in cluster_space.orders
        ]
        return RankInfo(
            parameters=sum(value.parameters for value in values),
            rank=sum(value.rank for value in values),
            aliases=sum(value.aliases for value in values),
        )
    block = cluster_space.block(order)
    orbit_indices = range(block.orbits.start, block.orbits.stop)
    groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for orbit_index in orbit_indices:
        for image, atoms in enumerate(image_atom_indices[orbit_index]):
            groups.setdefault(tuple(int(value) for value in atoms), []).append((orbit_index, image))
    aliases = sum(len(group) - 1 for group in groups.values())
    exclusive = {group[0][0] for group in groups.values() if len(group) == 1}
    if all(orbit_index in exclusive for orbit_index in orbit_indices):
        return RankInfo(
            block.parameters.stop - block.parameters.start,
            block.parameters.stop - block.parameters.start,
            aliases,
        )

    # Aliasing couples orbit parameter blocks. Rank each connected component
    # separately so an unrelated orbit never enlarges a dense rank workspace.
    parents = {index: index for index in orbit_indices}

    def root(index):
        while parents[index] != index:
            index = parents[index]
        return index

    for group in groups.values():
        first = root(group[0][0])
        for orbit_index, _ in group[1:]:
            other = root(orbit_index)
            if other != first:
                parents[other] = first
    components = {}
    for orbit_index in orbit_indices:
        components.setdefault(root(orbit_index), []).append(orbit_index)
    rank = 0
    for component, indices in components.items():
        keys = [key for key, group in groups.items() if root(group[0][0]) == component]
        row_locations = {key: i for i, key in enumerate(keys)}
        starts = {}
        width = 0
        for index in indices:
            starts[index] = width
            width += cluster_space.orbits[index].dimension
        tensor_dimension = 3**order
        # Different orbits occupy disjoint columns. Only images that reach
        # the same atom tuple and orbit can accumulate in the same entry.
        magnitudes = {
            index: max(
                (abs(int(v)) for v in cluster_space.orbits[index].exact_lattice_basis.flat),
                default=0,
            )
            for index in indices
        }
        for key in keys:
            totals = {}
            for index, image in groups[key]:
                orbit = cluster_space.orbits[index]
                rotation = cluster_space.symmetry.rotations[orbit.operations[image]]
                bound = tensor_action_bound(rotation, order, magnitudes[index])
                totals[index] = totals.get(index, 0) + bound
            for bound in totals.values():
                require_bound("folded tensor accumulation", bound)
        require_allocation("folded rank workspace", (len(keys) * tensor_dimension, width))
        matrix = np.zeros((len(keys) * tensor_dimension, width), dtype=np.int64)
        for index in indices:
            orbit = cluster_space.orbits[index]
            for image, atoms in enumerate(image_atom_indices[index]):
                basis = _tensor_action(
                    orbit.exact_lattice_basis,
                    cluster_space.symmetry.rotations[orbit.operations[image]],
                    orbit.permutations[image],
                )
                row_start = row_locations[tuple(int(v) for v in atoms)] * tensor_dimension
                accumulate_folded(matrix, basis, row_start, starts[index])
        rank += exact_rank(np.unique(matrix, axis=0))
    start = block.parameters.stop - block.parameters.start
    return RankInfo(parameters=start, rank=rank, aliases=aliases)


@njit(cache=True)
def accumulate_folded(out, basis, row_start, column_start):
    for i in range(basis.shape[0]):
        for j in range(basis.shape[1]):
            out[row_start + i, column_start + j] += basis[i, j]
