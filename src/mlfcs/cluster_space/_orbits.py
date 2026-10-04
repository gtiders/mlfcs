"""Space-group orbit actions and exact label deduplication."""

import numpy as np
from numba import njit

from mlfcs._arrays import integer_array, require_allocation
from mlfcs.cluster_space.models import Cluster
from mlfcs.core.symmetry import PrimitiveSymmetry


@njit(cache=True)
def cluster_hash(labels):
    # Bounded modular hashing; collisions are resolved by exact label equality.
    """Hash integer labels with bounded modular arithmetic; equality resolves collisions."""
    result = 0
    for value in labels.flat:
        result = (result * 65521 + value % 2147483647) % 2147483647
    return result


@njit(cache=True)
def registry_contains(labels, keys, used):
    """Probe exact labels in an open-addressed table with at least one unused slot."""
    index = cluster_hash(labels) % len(used)
    while used[index]:
        if np.all(keys[index] == labels):
            return True
        index = (index + 1) % len(used)
    return False


@njit(cache=True)
def registry_insert(values, keys, used):
    """Insert a batch into keys/used in place and return the number of newly occupied slots."""
    added = 0
    for labels in values:
        index = cluster_hash(labels) % len(used)
        while used[index] and not np.all(keys[index] == labels):
            index = (index + 1) % len(used)
        if not used[index]:
            keys[index] = labels
            used[index] = 1
            added += 1
    return added


@njit(cache=True)
def distinct_actions(values, representative):
    """Return first distinct action indices and all actions equal to the representative labels."""
    capacity = max(2, 2 * len(values))
    keys = np.empty((capacity, values.shape[1], 4), dtype=np.int64)
    used = np.zeros(capacity, dtype=np.uint8)
    indices = np.empty(len(values), dtype=np.int64)
    stabilizers = np.empty(len(values), dtype=np.int64)
    count = stable_count = 0
    for i in range(len(values)):
        if np.all(values[i] == representative):
            stabilizers[stable_count] = i
            stable_count += 1
        if not registry_contains(values[i], keys, used):
            registry_insert(values[i : i + 1], keys, used)
            indices[count] = i
            count += 1
    return indices[:count], stabilizers[:stable_count]


@njit(cache=True)
def transform_cluster(
    labels: np.ndarray,
    rotations: np.ndarray,
    site_permutations: np.ndarray,
    site_shifts: np.ndarray,
    axis_permutations: np.ndarray,
) -> np.ndarray:
    """Apply every affine site action and axis permutation, then re-anchor each image.

    Labels have shape (p, 4), rotations (s, 3, 3), site permutations (s, N),
    shifts (s, N, 3), and axis permutations (q, p). Return int64 (s*q, p, 4)
    with operation-major ordering. Each image subtracts its first translation.
    _integer_boundary must admit affine products and re-anchoring before entry.
    """
    operation_count = rotations.shape[0]
    permutation_count = axis_permutations.shape[0]
    order = labels.shape[0]
    transformed = np.empty((operation_count * permutation_count, order, 4), dtype=np.int64)
    for operation in range(operation_count):
        for permutation_index in range(permutation_count):
            target = operation * permutation_count + permutation_index
            for axis in range(order):
                source_axis = axis_permutations[permutation_index, axis]
                source_site = labels[source_axis, 0]
                transformed[target, axis, 0] = site_permutations[operation, source_site]
                for component in range(3):
                    value = site_shifts[operation, source_site, component]
                    for inner in range(3):
                        value += (
                            labels[source_axis, inner + 1] * rotations[operation, component, inner]
                        )
                    transformed[target, axis, component + 1] = value
            for component in range(3):
                origin = transformed[target, 0, component + 1]
                for axis in range(order):
                    transformed[target, axis, component + 1] -= origin
    return transformed


class ClusterRegistry:
    """Ephemeral per-order deduplication buffers, grown outside compiled loops."""

    def __init__(self, order):
        """Allocate an empty per-order hash table with sixteen slots."""
        self.order = order
        self.count = 0
        require_allocation("orbit registry", (16, order, 4))
        self.keys = np.empty((16, order, 4), dtype=np.int64)
        self.used = np.zeros(16, dtype=np.uint8)

    def contains(self, labels):

        """Test exact membership without modifying the registry."""
        return registry_contains(labels, self.keys, self.used)

    def update(self, labels):

        """Grow and rehash outside Numba as needed, then insert labels and update the count."""
        capacity = len(self.used)
        while 2 * (self.count + len(labels)) >= capacity:
            capacity *= 2
        if capacity != len(self.used):
            require_allocation("orbit registry", (capacity, self.order, 4))
            keys = np.empty((capacity, self.order, 4), dtype=np.int64)
            used = np.zeros(capacity, dtype=np.uint8)
            registry_insert(self.keys[self.used != 0], keys, used)
            self.keys, self.used = keys, used
        self.count += registry_insert(labels, self.keys, self.used)


def _integer_boundary(
    labels: tuple[tuple[int, int, int, int], ...], symmetry: PrimitiveSymmetry
) -> np.ndarray:
    """Prove that one cluster action fits int64, without a numerical tolerance."""
    array = integer_array(labels, name="primitive cluster labels")
    from mlfcs.core.symmetry import prove_site_actions

    if array.ndim != 2 or array.shape[1] != 4 or not len(array):
        raise ValueError("cluster labels must have shape (order,4) with positive order")
    if np.any(array[:, 0] < 0) or np.any(array[:, 0] >= symmetry.site_permutations.shape[1]):
        raise ValueError("cluster label site lies outside primitive motif")
    prove_site_actions(
        array[:, 1:], symmetry.rotations, symmetry.site_shifts[:, array[:, 0], :], reanchor=True
    )
    return array


def _cluster_key(values: np.ndarray) -> tuple[int, ...]:
    """Flatten one cluster into a Python-integer tuple for reference/introspection output."""
    return tuple(int(value) for value in values.reshape(-1))


def _orbit_actions(
    cluster: Cluster,
    symmetry: PrimitiveSymmetry,
    permutation_values: tuple[tuple[int, ...], ...],
    permutation_array: np.ndarray,
    *,
    return_keys: bool = True,
) -> tuple[
    Cluster,
    tuple[Cluster, ...],
    np.ndarray,
    np.ndarray,
    tuple[tuple[np.ndarray, tuple[int, ...]], ...],
    tuple[tuple[int, ...], ...],
]:
    """Return unique images, their tensor actions and representative stabilizers.

    The input cluster remains the representative. Enumerate space operations
    then axis permutations, retaining the first action for each exact image.
    Return (representative, clusters, operations, permutations, stabilizers, keys).
    Keys are optional introspection data; production deduplication uses buffers.
    Local integer and allocation checks precede the compiled transformations.
    """
    labels = _integer_boundary(cluster.labels, symmetry)
    require_allocation("orbit actions", (symmetry.size * len(permutation_values), cluster.order, 4))
    transformed = transform_cluster(
        labels,
        symmetry.rotations,
        symmetry.site_permutations,
        symmetry.site_shifts,
        permutation_array,
    )
    representative = cluster
    require_allocation(
        "orbit action deduplication", (max(2, 2 * len(transformed)), cluster.order, 4)
    )
    indices, stabilizer_indices = distinct_actions(transformed, labels)
    permutation_count = len(permutation_values)
    clusters = tuple(Cluster.from_labels(transformed[index]) for index in indices)
    operations = np.asarray(indices // permutation_count, dtype=np.int64)
    permutations = permutation_array[indices % permutation_count]
    stabilizers = tuple(
        (
            symmetry.rotations[index // permutation_count],
            permutation_values[index % permutation_count],
        )
        for index in stabilizer_indices
    )
    if not stabilizers:
        raise RuntimeError("the primitive symmetry group does not contain the identity action")
    # Keys are an introspection/reference result, not the production registry.
    keys = tuple(_cluster_key(transformed[index]) for index in indices) if return_keys else ()
    return representative, clusters, operations, permutations, stabilizers, keys
