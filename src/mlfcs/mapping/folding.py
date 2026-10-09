"""Assess structural parameter aliasing induced by supercell folding.

Primitive cluster images that fold onto the same ordered tuple of supercell
atoms are summed into the same rows of an integer linear map. The rank of
that map determines which primitive force-constant parameter directions
remain distinguishable in a given supercell, independently of any training
data.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from mlfcs.foundation.arrays import require_allocation, require_bound
from mlfcs.foundation.errors import AliasingError, RankError
from mlfcs.foundation.integer import as_int64_matrix, echelon, prime_stream, to_python_rows
from mlfcs.foundation.tensors import apply_tensor_action, tensor_action_bound


def _hadamard_minor_bound(matrix: np.ndarray, size: int) -> int:
    """Bound the absolute value of every ``size`` minor by Hadamard's inequality.

    The bound is the product of the ``size`` largest Euclidean row-norm bounds
    and is used to certify rational rank from modular rank calculations.
    """
    if size <= 0:
        return 1
    rows = to_python_rows(matrix)
    row_bounds = sorted(
        (math.isqrt(sum(value * value for value in row)) + 1 for row in rows), reverse=True
    )
    return math.prod(row_bounds[:size])


def _matrix_rank(matrix: object) -> int:
    """Certify the rational rank of an integer matrix using modular arithmetic.

    For each tested prime ``p``, the rank over ``F_p`` gives a lower bound on
    the rational rank. If a full-rank modular witness is found, the rank is
    settled immediately. Otherwise, a product of row-norm bounds, each at
    least one, bounds every minor up to the maximum possible order. Once the
    product of tested primes exceeds this bound, every minor larger than the
    best modular rank must be zero over the integers, certifying that rank.

    Returns
    -------
    int
        Exact rank over the rationals.

    Raises
    ------
    RankError
        If the available prime stream is exhausted before the rank can be
        certified.
    """
    values = as_int64_matrix(matrix)
    target = min(values.shape)
    if target == 0:
        return 0
    minor_bound = _hadamard_minor_bound(values, target)
    prime_product, best_rank = 1, 0
    for prime in prime_stream():
        _, pivot_columns, _ = echelon(values, prime)
        best_rank = max(best_rank, len(pivot_columns))
        prime_product *= prime
        if best_rank == target or prime_product > minor_bound:
            return best_rank
    raise RankError("prime stream exhausted before determining folded matrix rank")


@dataclass(frozen=True, slots=True)
class RankInfo:
    """Structural identifiability of primitive parameters in a supercell.

    ``parameters`` is the number of primitive force-constant parameters before
    folding, while ``rank`` is the dimension that remains distinguishable
    after all symmetry images are mapped onto supercell atom tuples. The rank
    is exact over the rationals. ``aliases`` counts repeated folded images and
    is only a diagnostic; it is not generally equal to the rank loss.
    """

    parameters: int
    rank: int
    aliases: int

    @property
    def nullity(self) -> int:
        """Number of primitive parameters lost by supercell folding."""
        return self.parameters - self.rank

    @property
    def full(self) -> bool:
        """Whether folding preserves every primitive parameter direction."""
        return self.rank == self.parameters

    def require_full(self) -> None:
        """Require the supercell to preserve all primitive parameter directions."""
        if not self.full:
            raise AliasingError(
                f"supercell realization has rank {self.rank} for {self.parameters} "
                f"parameters (nullity {self.nullity})"
            )


def group_folded_images(image_atom_indices, orbit_indices):
    """Group cluster images that fold onto the same ordered supercell atom tuple.

    Each orbit image is represented by the ordered tuple of supercell atoms
    reached by its tensor slots. Images with identical tuples accumulate into
    the same rows of the folded linear map and may therefore alias primitive
    parameter directions.

    Returns
    -------
    dict
        Insertion-ordered mapping from atom tuples to ``(orbit, image)`` pairs.
    """
    groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for orbit_index in orbit_indices:
        for image, atoms in enumerate(image_atom_indices[orbit_index]):
            groups.setdefault(tuple(int(value) for value in atoms), []).append((orbit_index, image))
    return groups


def folded_rank(cluster_space, image_atom_indices, order=None):
    """Compute the structural rank of primitive parameters after supercell folding.

    Each symmetry image of a primitive orbit contributes a transformed tensor
    basis to the ordered tuple of supercell atoms onto which that image folds.
    Contributions reaching the same tuple are summed, producing an integer
    linear map from primitive force-constant parameters to folded supercell
    tensor components.

    The rational rank of this map is the number of primitive parameter
    directions that remain distinguishable in the chosen supercell. Its
    nullity therefore measures parameter directions lost purely through
    periodic folding, independently of displacement patterns or training data.

    Parameters
    ----------
    cluster_space
        Primitive force-constant parameter space.
    image_atom_indices
        For each global orbit, integer arrays of shape
        ``(n_images, tensor_order)`` giving the supercell atom tuple of every
        symmetry image.
    order
        Optional force-constant order. If omitted, ranks are summed over all
        included orders.

    Returns
    -------
    RankInfo
        Number of primitive parameters, exact folded rank, and image-alias
        count.
    """
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
    groups = group_folded_images(image_atom_indices, orbit_indices)
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
        """Find the current orbit-component representative without path compression."""
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
                (abs(int(v)) for v in cluster_space.orbits[index].lattice_basis.flat),
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
                basis = apply_tensor_action(
                    orbit.lattice_basis,
                    cluster_space.symmetry.rotations[orbit.operations[image]],
                    orbit.permutations[image],
                )
                row_start = row_locations[tuple(int(v) for v in atoms)] * tensor_dimension
                accumulate_folded(matrix, basis, row_start, starts[index])
        rank += _matrix_rank(np.unique(matrix, axis=0))
    start = block.parameters.stop - block.parameters.start
    return RankInfo(parameters=start, rank=rank, aliases=aliases)


@njit(cache=True)
def accumulate_folded(out, basis, row_start, column_start):
    """Accumulate one transformed orbit basis into the folded linear map.

    ``basis`` holds the lattice-coordinate tensor basis of one orbit image; it
    is added to the row block of its folded supercell atom tuple and to the
    parameter columns of its primitive orbit. ``folded_rank`` validates the
    per-tuple accumulation bounds before entry.
    """
    for i in range(basis.shape[0]):
        for j in range(basis.shape[1]):
            out[row_start + i, column_start + j] += basis[i, j]
