"""Primitive-cell interaction orbits with exact integer translations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import spglib
from ase import Atoms
from ase.neighborlist import neighbor_list

from mlfcs.core.geometry import _unique_distances
from mlfcs.core.orbits import (
    ClusterOrbit,
    OrbitImage,
    OrbitSpace,
    TensorAction,
    _independent_basis_rows,
    _label_symmetric_basis,
    _null_space_from_gram,
)


@dataclass(frozen=True, order=True, slots=True)
class InteractionKey:
    """One anchored primitive interaction with exact lattice translations."""

    sites: tuple[int, ...]
    translations: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if len(self.sites) < 2:
            raise ValueError("an interaction must contain at least two indices")
        if len(self.translations) != len(self.sites) - 1:
            raise ValueError("an anchored interaction requires one fewer translations than sites")
        if any(len(value) != 3 for value in self.translations):
            raise ValueError("interaction translations must be integer 3-vectors")

    @property
    def order(self) -> int:
        return len(self.sites)

    @property
    def labels(self) -> tuple[tuple[int, int, int, int], ...]:
        return (
            (self.sites[0], 0, 0, 0),
            *(
                (site, int(translation[0]), int(translation[1]), int(translation[2]))
                for site, translation in zip(self.sites[1:], self.translations, strict=True)
            ),
        )

    @classmethod
    def from_labels(cls, labels) -> InteractionKey:
        values = tuple(tuple(int(value) for value in label) for label in labels)
        if not values:
            raise ValueError("an interaction key cannot be empty")
        origin = np.asarray(values[0][1:], dtype=np.int64)
        translations = tuple(
            tuple(int(value) for value in np.asarray(label[1:], dtype=np.int64) - origin)
            for label in values[1:]
        )
        return cls(tuple(label[0] for label in values), translations)


@dataclass(frozen=True, slots=True)
class PrimitiveSymmetryOperations:
    rotations: np.ndarray
    translations: np.ndarray
    cartesian_rotations: np.ndarray
    site_permutations: np.ndarray
    site_shifts: np.ndarray
    symbol: str

    @classmethod
    def from_atoms(cls, primitive: Atoms, *, symprec: float) -> PrimitiveSymmetryOperations:
        cell = (
            np.asarray(primitive.cell),
            primitive.get_scaled_positions(),
            primitive.numbers,
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is None:
            raise ValueError("spglib could not determine the primitive crystal symmetry")
        rotations = np.asarray(dataset.rotations, dtype=np.int32)
        translations = np.asarray(dataset.translations, dtype=float)
        lattice = np.asarray(primitive.cell)
        inverse = np.linalg.inv(lattice)
        cartesian = np.asarray([inverse @ rotation.T @ lattice for rotation in rotations])
        scaled = primitive.get_scaled_positions(wrap=False)
        permutations_array = np.empty((len(rotations), len(primitive)), dtype=np.int32)
        shifts = np.empty((len(rotations), len(primitive), 3), dtype=np.int32)
        for operation, (rotation, translation) in enumerate(
            zip(rotations, translations, strict=True)
        ):
            transformed = scaled @ rotation.T + translation
            for site, position in enumerate(transformed):
                candidates = np.flatnonzero(primitive.numbers == primitive.numbers[site])
                differences = position - scaled[candidates]
                integers = np.rint(differences).astype(np.int32)
                residuals = np.linalg.norm((differences - integers) @ lattice, axis=1)
                selected = np.flatnonzero(residuals < symprec * 10.0)
                if len(selected) != 1:
                    raise ValueError(
                        f"symmetry operation {operation} maps primitive site {site} "
                        f"to {len(selected)} sites"
                    )
                location = int(selected[0])
                permutations_array[operation, site] = int(candidates[location])
                shifts[operation, site] = integers[location]
        return cls(
            rotations,
            translations,
            cartesian,
            permutations_array,
            shifts,
            dataset.international.strip(),
        )

    @property
    def size(self) -> int:
        return len(self.rotations)

    def transform_label(
        self, operation: int, label: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        site = int(label[0])
        translation = np.asarray(label[1:], dtype=np.int64)
        transformed = translation @ self.rotations[operation].T + self.site_shifts[operation, site]
        return (
            int(self.site_permutations[operation, site]),
            *(int(value) for value in transformed),
        )


@dataclass(frozen=True, slots=True)
class PrimitiveOrbitImage:
    key: InteractionKey
    action: TensorAction


@dataclass(frozen=True, slots=True)
class PrimitiveInteractionOrbit:
    representative: InteractionKey
    basis: np.ndarray
    pivots: np.ndarray
    images: tuple[PrimitiveOrbitImage, ...]

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True, slots=True)
class PrimitiveInteractionSpace:
    primitive: Atoms
    order: int
    cutoff: float
    max_body_order: int | None
    symmetry: PrimitiveSymmetryOperations
    orbits: tuple[PrimitiveInteractionOrbit, ...]


class InteractionAliasingError(ValueError):
    """A finite reference folds distinct primitive interactions together."""


def validate_realization_identifiability(
    space: PrimitiveInteractionSpace,
    index,
    *,
    tolerance: float = 1e-10,
) -> None:
    """Reject a finite reference that cannot identify primitive parameters.

    The realization matrix is assembled in concrete IFC-component space.
    Its column graph normally separates into small independent components, so
    exact rank tests do not require a dense global matrix.
    """
    parameter_offsets = np.cumsum(
        [0, *(orbit.dimension for orbit in space.orbits)], dtype=np.int64
    )
    rows: dict[tuple[tuple[int, ...], int], dict[int, float]] = {}
    for orbit_index, orbit in enumerate(space.orbits):
        offset = int(parameter_offsets[orbit_index])
        for image in orbit.images:
            cluster = _realize_key(image.key, index)
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

    for values in rows.values():
        columns = tuple(values)
        for column in columns[1:]:
            union(columns[0], column)
    components: dict[int, list[int]] = {}
    for column in range(n_parameters):
        components.setdefault(find(column), []).append(column)

    row_items = tuple(rows.items())
    for columns in components.values():
        selected = set(columns)
        component_rows = [
            values for _key, values in row_items if selected.intersection(values)
        ]
        matrix = np.asarray(
            [[values.get(column, 0.0) for column in columns] for values in component_rows],
            dtype=float,
        )
        rank = int(np.linalg.matrix_rank(matrix, tol=tolerance))
        if rank != len(columns):
            affected = [
                orbit.representative
                for orbit_index, orbit in enumerate(space.orbits)
                if any(
                    int(parameter_offsets[orbit_index]) <= column
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


def resolve_primitive_cutoff(primitive: Atoms, cutoff: float) -> float:
    """Resolve a required distance or neighbor-shell cutoff on the infinite lattice."""
    if cutoff is None:
        raise ValueError("cutoff must be explicit for a primitive interaction space")
    value = float(cutoff)
    if value > 0:
        return value
    if not value.is_integer():
        raise ValueError("cutoff must be a positive distance or negative integer shell")
    shell = -int(value)
    if shell < 1:
        raise ValueError("neighbor shell must be positive")
    radius = max(float(np.min(np.linalg.norm(np.asarray(primitive.cell), axis=1))), 1.0)
    for _ in range(16):
        first, _second, distances = neighbor_list("ijd", primitive, radius, self_interaction=False)
        shells = []
        for site in range(len(primitive)):
            try:
                shells.append(_unique_distances(distances[first == site]))
            except ValueError:
                shells.append([])
        if all(len(values) > shell for values in shells):
            return float(max((values[shell - 1] + values[shell]) / 2.0 for values in shells))
        radius *= 2.0
    raise RuntimeError("could not resolve the requested primitive neighbor shell")


def build_primitive_interaction_space(
    primitive: Atoms,
    *,
    order: int,
    cutoff: float,
    max_body_order: int | None,
    symprec: float,
    tolerance: float = 1e-9,
) -> PrimitiveInteractionSpace:
    """Enumerate and symmetry-reduce exact primitive-lattice interactions."""
    if order < 2:
        raise ValueError("order must be at least two")
    if max_body_order is not None and not 1 <= max_body_order <= order:
        raise ValueError("max_body_order must be between 1 and order")
    primitive = primitive.copy()
    primitive.wrap()
    radius = resolve_primitive_cutoff(primitive, cutoff)
    symmetry = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=symprec)
    neighbors = _primitive_neighbors(primitive, radius)
    axis_permutations = tuple(permutations(range(order)))
    seen: set[InteractionKey] = set()
    orbits: list[PrimitiveInteractionOrbit] = []
    for anchor in range(len(primitive)):
        candidates = neighbors[anchor]
        for tail in _compatible_tails(candidates, order - 1, primitive, radius):
            key = InteractionKey.from_labels(((anchor, 0, 0, 0), *tail))
            if max_body_order is not None and len(set(key.labels)) > max_body_order:
                continue
            representative = _canonical_key(key, symmetry)
            if representative in seen:
                continue
            seen.add(representative)
            label_basis = _label_symmetric_basis(representative.labels)
            gram = np.zeros((label_basis.shape[1],) * 2)
            action_by_image: dict[InteractionKey, TensorAction] = {}
            for operation in range(symmetry.size):
                transformed = tuple(
                    symmetry.transform_label(operation, label) for label in representative.labels
                )
                for permutation in axis_permutations:
                    image = InteractionKey.from_labels(transformed[axis] for axis in permutation)
                    action = TensorAction(
                        symmetry.cartesian_rotations[operation].T,
                        permutation,
                        order,
                    )
                    action_by_image.setdefault(image, action)
                    if image == representative:
                        constraint = action.apply_columns(label_basis) - label_basis
                        gram += constraint.T @ constraint
            reduced_basis, _ = _null_space_from_gram(gram, tolerance)
            basis = label_basis @ reduced_basis
            if basis.shape[1] == 0:
                continue
            pivots = _independent_basis_rows(basis, tolerance)
            basis = basis @ np.linalg.inv(basis[pivots])
            images = tuple(
                PrimitiveOrbitImage(image, action)
                for image, action in sorted(action_by_image.items())
            )
            orbits.append(PrimitiveInteractionOrbit(representative, basis, pivots, images))
    return PrimitiveInteractionSpace(
        primitive,
        order,
        radius,
        max_body_order,
        symmetry,
        tuple(orbits),
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


def _primitive_neighbors(primitive: Atoms, cutoff: float):
    first, second, shifts, distances = neighbor_list(
        "ijSd", primitive, cutoff, self_interaction=True
    )
    result: list[list[tuple[int, int, int, int]]] = [[] for _ in primitive]
    for anchor, site, shift, distance in zip(first, second, shifts, distances, strict=True):
        if float(distance) < cutoff:
            result[int(anchor)].append((int(site), *(int(value) for value in shift)))
    return [tuple(sorted(set(values))) for values in result]


def _compatible_tails(candidates, length, primitive: Atoms, cutoff: float):
    positions = primitive.get_scaled_positions(wrap=False)
    cell = np.asarray(primitive.cell)
    prefix: list[tuple[int, int, int, int]] = []

    def coordinate(label):
        return (positions[label[0]] + np.asarray(label[1:], dtype=float)) @ cell

    def extend(start):
        if len(prefix) == length:
            yield tuple(prefix)
            return
        for location in range(start, len(candidates)):
            candidate = candidates[location]
            point = coordinate(candidate)
            if all(np.linalg.norm(point - coordinate(previous)) < cutoff for previous in prefix):
                prefix.append(candidate)
                yield from extend(location)
                prefix.pop()

    yield from extend(0)


def _canonical_key(
    key: InteractionKey, symmetry: PrimitiveSymmetryOperations
) -> InteractionKey:
    best: InteractionKey | None = None
    for operation in range(symmetry.size):
        transformed = tuple(
            symmetry.transform_label(operation, label) for label in key.labels
        )
        for anchor in range(key.order):
            ordered = (transformed[anchor],) + transformed[:anchor] + transformed[anchor + 1 :]
            candidate = InteractionKey.from_labels((ordered[0], *sorted(ordered[1:])))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best


__all__ = [
    "InteractionAliasingError",
    "InteractionKey",
    "PrimitiveInteractionOrbit",
    "PrimitiveInteractionSpace",
    "PrimitiveOrbitImage",
    "PrimitiveSymmetryOperations",
    "build_primitive_interaction_space",
    "realize_orbit_space",
    "resolve_primitive_cutoff",
    "validate_realization_identifiability",
]
