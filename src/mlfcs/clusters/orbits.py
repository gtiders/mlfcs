from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from ase import Atoms

from mlfcs.core.symmetry import SymmetryOperations
from mlfcs.structure.geometry import PeriodicGeometry, PeriodicIndex


@dataclass(frozen=True, slots=True)
class TensorAction:
    """Host-side Cartesian rotation followed by an IFC-axis permutation.

    Orbit enumeration, constraint construction, and finite-difference
    reconstruction invoke this operation frequently on small, irregular
    tensors.  Keeping it in NumPy avoids device dispatch and transfer costs;
    the fitting-only JAX feature kernels implement their own batched form.
    """

    rotation: np.ndarray
    permutation: tuple[int, ...]
    order: int

    def apply(self, tensor: np.ndarray) -> np.ndarray:
        return _apply_action_tensor_numpy(self, np.asarray(tensor, dtype=float))

    def apply_flat(self, values: np.ndarray) -> np.ndarray:
        tensor = np.asarray(values).reshape((3,) * self.order)
        return self.apply(tensor).reshape(-1)

    def apply_columns(self, values: np.ndarray) -> np.ndarray:
        return _apply_action_columns_numpy(self, values)

    def as_matrix(self) -> np.ndarray:
        return tensor_action_matrix(self.rotation, self.permutation, self.order)


@dataclass(frozen=True, slots=True)
class OrbitImage:
    cluster: tuple[int, ...]
    action: TensorAction


@dataclass(frozen=True, slots=True)
class ClusterOrbit:
    representative: tuple[int, ...]
    basis: np.ndarray
    pivots: np.ndarray
    images: tuple[OrbitImage, ...]

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True, slots=True)
class OrbitSpace:
    order: int
    orbits: tuple[ClusterOrbit, ...]
    cutoff: float
    max_body_order: int | None = None

    @property
    def displacement_keys(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        keys: set[tuple[tuple[int, int], ...]] = set()
        for orbit in self.orbits:
            for flat_component in orbit.pivots:
                components = np.unravel_index(int(flat_component), (3,) * self.order)
                key = tuple(
                    (orbit.representative[axis], int(components[axis]))
                    for axis in range(self.order - 1)
                )
                keys.add(key)
        return tuple(sorted(keys))


def cluster_invariant_dimension(
    cluster: tuple[int, ...],
    index: PeriodicIndex,
    symmetry: SymmetryOperations,
    *,
    tolerance: float = 1e-9,
) -> int:
    """Return the symmetry-allowed Cartesian tensor dimension of one cluster.

    This deliberately ignores cutoff and body-order support.  It is used to
    distinguish a genuinely omitted interaction from a cluster whose tensor
    vanishes identically under its site stabilizer.
    """
    representative = _canonical_cluster(cluster, index, symmetry)
    label_basis = _label_symmetric_basis(representative)
    constraint_gram = np.zeros((label_basis.shape[1],) * 2)
    for operation in range(symmetry.size):
        transformed = tuple(
            int(symmetry.atom_permutations[operation, atom]) for atom in representative
        )
        for axis_permutation in permutations(range(len(representative))):
            candidate = index.anchor(tuple(transformed[axis] for axis in axis_permutation))
            if candidate == representative:
                action = TensorAction(
                    symmetry.cartesian_rotations[operation].T,
                    axis_permutation,
                    len(representative),
                )
                constraint = _apply_action_columns_numpy(action, label_basis) - label_basis
                constraint_gram += constraint.T @ constraint
    reduced_basis, _ = _null_space_from_gram(constraint_gram, tolerance)
    return int(reduced_basis.shape[1])


def build_orbit_space(
    supercell: Atoms,
    index: PeriodicIndex,
    symmetry: SymmetryOperations,
    *,
    order: int,
    cutoff: float,
    max_body_order: int | None = None,
    tolerance: float = 1e-9,
) -> OrbitSpace:
    """Enumerate cutoff clusters and reduce them by permutation and space group."""
    if order < 2:
        raise ValueError("order must be at least two")
    if max_body_order is not None and not 1 <= max_body_order <= order:
        raise ValueError("max_body_order must be between 1 and order")
    anchors = np.asarray([index.representative(site) for site in range(index.n_primitive)])
    distances, tail_compatibility = _joint_periodic_cluster_geometry(supercell, anchors, cutoff)
    neighbors = [
        np.flatnonzero(distances[atom] < cutoff).tolist() for atom in range(index.n_primitive)
    ]
    axis_permutations = tuple(permutations(range(order)))
    seen_representatives: set[tuple[int, ...]] = set()
    orbits: list[ClusterOrbit] = []

    for site, first in enumerate(anchors):
        for tail in _compatible_sorted_tails(neighbors[site], order - 1, tail_compatibility[site]):
            cluster = (first, *tail)
            if max_body_order is not None and len(set(cluster)) > max_body_order:
                continue
            # The cutoff support is tested on the generated candidate, but its
            # global symmetry representative need not itself be generated by
            # that anchored support test.  Canonicalize every accepted seed
            # and deduplicate representatives; requiring seed == canonical
            # would silently discard complete orbits at periodic boundaries.
            representative = _canonical_cluster(cluster, index, symmetry)
            if representative in seen_representatives:
                continue
            seen_representatives.add(representative)
            action_by_image: dict[tuple[int, ...], TensorAction] = {}
            # Impose exchanges of axes carrying the same atomic label up
            # front.  This is the label-symmetric tensor basis used by
            # hiPhive: repeated-site FC6 clusters can have tens rather than
            # 729 working components.  Space-group stabilizers are then
            # accumulated directly in this reduced coordinate system.
            label_basis = _label_symmetric_basis(representative)
            constraint_gram = np.zeros((label_basis.shape[1],) * 2)
            for operation in range(symmetry.size):
                transformed = tuple(
                    int(symmetry.atom_permutations[operation, atom]) for atom in representative
                )
                for axis_permutation in axis_permutations:
                    candidate = index.anchor(tuple(transformed[axis] for axis in axis_permutation))
                    action = TensorAction(
                        symmetry.cartesian_rotations[operation].T,
                        axis_permutation,
                        order,
                    )
                    # Two mappings to one image differ by a representative
                    # stabilizer, so either acts identically on the invariant basis.
                    action_by_image.setdefault(candidate, action)
                    if candidate == representative:
                        constraint = _apply_action_columns_numpy(action, label_basis) - label_basis
                        constraint_gram += constraint.T @ constraint
            reduced_basis, _ = _null_space_from_gram(constraint_gram, tolerance)
            basis = label_basis @ reduced_basis
            if basis.shape[1] == 0:
                continue
            pivots = _independent_basis_rows(basis, tolerance)
            # Express the invariant subspace in its deterministic pivot-value
            # coordinates.  Besides matching the finite-difference contract
            # (the parameters are actual selected tensor components), this
            # avoids carrying the arbitrary normalization of the label basis.
            basis = basis @ np.linalg.inv(basis[pivots])
            orbit_images = tuple(
                OrbitImage(key, action) for key, action in sorted(action_by_image.items())
            )
            orbits.append(ClusterOrbit(representative, basis, pivots, orbit_images))
    return OrbitSpace(order, tuple(orbits), cutoff, max_body_order)


def _compatible_sorted_tails(
    neighbors: list[int],
    length: int,
    compatibility: np.ndarray,
):
    """Yield compatible neighbor multisets with prefix pruning.

    The old ordered Cartesian product generated every permutation and tested
    pair compatibility only after a complete tail existed.  IFC label
    permutation symmetry means that one nondecreasing tail is sufficient for
    orbit discovery.  Checking a new atom against the current prefix prevents
    an invalid partial clique from spawning any descendants.
    """
    values = tuple(sorted(int(atom) for atom in neighbors))
    prefix: list[int] = []

    def extend(start: int):
        if len(prefix) == length:
            yield tuple(prefix)
            return
        for position in range(start, len(values)):
            atom = values[position]
            if all(compatibility[atom, previous] for previous in prefix):
                prefix.append(atom)
                yield from extend(position)
                prefix.pop()

    yield from extend(0)


def _canonical_cluster(
    cluster: tuple[int, ...],
    index: PeriodicIndex,
    symmetry: SymmetryOperations,
) -> tuple[int, ...]:
    """Return the smallest anchored, tail-sorted space-group image.

    Only the choice of anchor is required here.  Sorting the remaining atoms
    represents all their label permutations, reducing the canonical check
    from ``order!`` mappings per symmetry operation to at most ``order``.
    Full ordered images are constructed only for accepted orbit prototypes.
    """
    best: tuple[int, ...] | None = None
    for operation in range(symmetry.size):
        transformed = tuple(int(symmetry.atom_permutations[operation, atom]) for atom in cluster)
        for anchor_axis in range(len(cluster)):
            anchored = index.anchor(
                (transformed[anchor_axis],)
                + transformed[:anchor_axis]
                + transformed[anchor_axis + 1 :]
            )
            candidate = (anchored[0], *sorted(anchored[1:]))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best


def _label_symmetric_basis(cluster: tuple[int, ...]) -> np.ndarray:
    """Return an orthonormal basis invariant under equal-site axis swaps."""
    groups: list[np.ndarray] = []
    values = np.asarray(cluster)
    for atom in dict.fromkeys(cluster):
        positions = np.flatnonzero(values == atom)
        if len(positions) > 1:
            groups.append(positions)

    size = 3 ** len(cluster)
    if not groups:
        return np.eye(size)
    classes: dict[tuple[int, ...], list[int]] = {}
    for flat, component in enumerate(np.ndindex((3,) * len(cluster))):
        canonical = list(component)
        for positions in groups:
            ordered = sorted(canonical[int(position)] for position in positions)
            for position, direction in zip(positions, ordered, strict=True):
                canonical[int(position)] = direction
        classes.setdefault(tuple(canonical), []).append(flat)
    basis = np.zeros((size, len(classes)))
    for column, members in enumerate(classes.values()):
        basis[members, column] = 1.0 / np.sqrt(len(members))
    return basis


def _apply_action_columns_numpy(action: TensorAction, values: np.ndarray) -> np.ndarray:
    """Apply one tensor action without constructing a ``3**order`` square matrix.

    Orbit construction is host-side and calls this only for stabilizers.  A
    NumPy contraction over the already compressed label basis avoids the large
    XLA intermediates produced by vmapping hundreds of sixth-order tensors.
    """
    tensors = np.asarray(values, dtype=float).T.reshape((-1,) + (3,) * action.order)
    transformed = tensors
    for axis in range(action.order):
        transformed = np.tensordot(action.rotation, transformed, axes=((1,), (axis + 1,)))
        transformed = np.moveaxis(transformed, 0, axis + 1)
    axes = (0,) + tuple(axis + 1 for axis in action.permutation)
    return np.transpose(transformed, axes).reshape(len(values.T), -1).T


def _apply_action_tensor_numpy(action: TensorAction, tensor: np.ndarray) -> np.ndarray:
    """Apply one action to one tensor without a dense representation matrix."""
    if tensor.shape != (3,) * action.order:
        raise ValueError(f"expected tensor shape {(3,) * action.order}, got {tensor.shape}")
    transformed = tensor
    for axis in range(action.order):
        transformed = np.tensordot(action.rotation, transformed, axes=((1,), (axis,)))
        transformed = np.moveaxis(transformed, 0, axis)
    return np.transpose(transformed, action.permutation)


def _independent_basis_rows(basis: np.ndarray, tolerance: float) -> np.ndarray:
    """Select deterministic independent tensor components for reconstruction."""
    matrix = basis.T.copy()
    n_rows, n_columns = matrix.shape
    selected: list[int] = []
    pivot_row = 0
    threshold = tolerance * max(float(np.max(np.abs(matrix))), 1.0)
    # Prefer high flattened Cartesian components, matching the free-column
    # convention of the historical full-space RREF while operating on the
    # compressed label-symmetric basis.
    for column in range(n_columns - 1, -1, -1):
        if pivot_row == n_rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(matrix[pivot_row:, column])))
        if abs(matrix[candidate, column]) <= threshold:
            continue
        matrix[[pivot_row, candidate]] = matrix[[candidate, pivot_row]]
        matrix[pivot_row] /= matrix[pivot_row, column]
        for row in range(pivot_row + 1, n_rows):
            if abs(matrix[row, column]) > threshold:
                matrix[row] -= matrix[row, column] * matrix[pivot_row]
        selected.append(column)
        pivot_row += 1
    if len(selected) != basis.shape[1]:
        raise RuntimeError("failed to select independent invariant tensor components")
    return np.asarray(sorted(selected), dtype=np.int32)


def _joint_periodic_cluster_geometry(
    supercell: Atoms,
    anchors: np.ndarray,
    cutoff: float,
    *,
    degeneracy_tolerance: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cutoff geometry with the joint-image convention of the old package.

    Images of every tail atom are first restricted to those at the minimum
    distance from the primitive anchor. Two tail atoms are compatible when at
    least one pair of these anchor-minimum images is within the cutoff. The old
    code used a squared-distance tolerance of ``1e-4 nm**2``, equivalent to
    ``1e-2 angstrom**2``.
    """
    if np.isscalar(anchors):  # compatibility with the former private helper
        anchors = np.arange(int(anchors), dtype=np.int32)
    anchors = np.asarray(anchors, dtype=np.int32)
    n_supercell = len(supercell)
    positions = supercell.positions
    distances = np.empty((len(anchors), n_supercell), dtype=float)
    minimum_images: list[list[np.ndarray]] = []
    geometry = PeriodicGeometry(supercell.cell, supercell.pbc)
    for row, first in enumerate(anchors):
        delta = positions - positions[first]
        _, lengths = geometry.mic(delta)
        distances[row] = lengths
        images = []
        for atom in range(n_supercell):
            nearest, _ = geometry.closest_images(delta[atom])
            # Historical orbit cutoff semantics use a comparatively generous
            # squared-distance degeneracy tolerance.  Preserve that tolerance
            # while sourcing the images from the common reduced-lattice core.
            squared = np.sum(nearest**2, axis=1)
            images.append(nearest[np.abs(squared - squared.min()) < degeneracy_tolerance])
        minimum_images.append(images)

    compatible = np.zeros((len(anchors), n_supercell, n_supercell), dtype=bool)
    cutoff_squared = cutoff * cutoff
    for first in range(len(anchors)):
        neighbors = np.flatnonzero(distances[first] < cutoff)
        compatible[first, neighbors, neighbors] = True
        for left_index, left in enumerate(neighbors):
            left_images = minimum_images[first][left]
            for right in neighbors[left_index + 1 :]:
                right_images = minimum_images[first][right]
                squared = np.sum(
                    (left_images[:, None, :] - right_images[None, :, :]) ** 2,
                    axis=2,
                )
                value = bool(np.any(squared < cutoff_squared))
                compatible[first, left, right] = value
                compatible[first, right, left] = value
    return distances, compatible


def tensor_action_matrix(
    rotation: np.ndarray,
    axis_permutation: tuple[int, ...],
    order: int,
) -> np.ndarray:
    """Return the Cartesian tensor representation of a symmetry action."""
    size = 3**order
    action = TensorAction(np.asarray(rotation, dtype=float), axis_permutation, order)
    return _apply_action_columns_numpy(action, np.eye(size))


def permute_tensor_action(
    action: np.ndarray,
    axis_permutation: tuple[int, ...],
    order: int,
) -> np.ndarray:
    """Permute output axes of a flattened tensor action."""
    size = 3**order
    by_input = action.T.reshape((size,) + (3,) * order)
    axes = (0,) + tuple(axis + 1 for axis in axis_permutation)
    return np.transpose(by_input, axes).reshape(size, size).T


def _null_space_from_gram(gram: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Find null(C) from the small Gram matrix C.T @ C.

    The direct constraint matrix may have tens of thousands of rows for a
    fourth-order high-symmetry cluster. Its Gram matrix is always at most
    81x81, preventing LAPACK from allocating a huge left-singular-vector
    matrix.
    """
    matrix = gram.copy()
    n_rows, n_columns = matrix.shape
    pivot_columns: list[int] = []
    pivot_row = 0
    threshold = tolerance * max(float(np.max(np.abs(matrix))), 1.0)
    for column in range(n_columns):
        if pivot_row == n_rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(matrix[pivot_row:, column])))
        if abs(matrix[candidate, column]) <= threshold:
            continue
        if candidate != pivot_row:
            matrix[[pivot_row, candidate]] = matrix[[candidate, pivot_row]]
        matrix[pivot_row] /= matrix[pivot_row, column]
        for row in range(n_rows):
            if row != pivot_row and abs(matrix[row, column]) > threshold:
                matrix[row] -= matrix[row, column] * matrix[pivot_row]
        matrix[np.abs(matrix) < threshold] = 0.0
        pivot_columns.append(column)
        pivot_row += 1

    independent = np.asarray(
        [column for column in range(n_columns) if column not in pivot_columns],
        dtype=np.int32,
    )
    basis = np.zeros((n_columns, len(independent)), dtype=float)
    for basis_column, free_column in enumerate(independent):
        basis[free_column, basis_column] = 1.0
        for row, dependent_column in enumerate(pivot_columns):
            basis[dependent_column, basis_column] = -matrix[row, free_column]
    basis[np.abs(basis) < tolerance] = 0.0
    return basis, independent


__all__ = [
    "ClusterOrbit",
    "OrbitImage",
    "OrbitSpace",
    "TensorAction",
    "_canonical_cluster",
    "_compatible_sorted_tails",
    "_joint_periodic_cluster_geometry",
    "_label_symmetric_basis",
    "build_orbit_space",
    "cluster_invariant_dimension",
    "permute_tensor_action",
    "tensor_action_matrix",
]
