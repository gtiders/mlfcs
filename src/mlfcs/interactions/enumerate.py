"""Cutoff resolution and primitive interaction-orbit enumeration."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from ase import Atoms
from ase.geometry import minkowski_reduce
from ase.neighborlist import neighbor_list

from mlfcs.interactions.keys import InteractionKey
from mlfcs.interactions.orbits import (
    PrimitiveInteractionOrbit,
    PrimitiveInteractionSpace,
    PrimitiveOrbitImage,
)
from mlfcs.interactions.tensors import (
    TensorAction,
    _independent_basis_rows,
    _label_symmetric_basis,
    _null_space_from_gram,
)
from mlfcs.structure.periodic_geometry import unique_periodic_distances
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations


def resolve_primitive_cutoff(
    primitive: Atoms,
    cutoff: float | None,
    *,
    reference: Atoms | None = None,
) -> float:
    """Resolve a distance, neighbor shell, or maximum finite-cell cutoff.

    ``None`` selects the largest radius for which each reference atom index
    has at most one periodic image in every neighbor list.  A 0.01 Å margin
    keeps the selected radius strictly away from the first ambiguous shell.
    """
    if cutoff is None:
        if reference is None:
            raise ValueError("cutoff=None requires an explicit reference supercell")
        reduced_cell, _operation = minkowski_reduce(reference.cell, pbc=reference.pbc)
        periodic_lengths = np.linalg.norm(
            np.asarray(reduced_cell)[np.asarray(reference.pbc)], axis=1
        )
        if not len(periodic_lengths):
            raise ValueError("cutoff=None requires at least one periodic direction")
        # No unambiguous cutoff can reach beyond the shortest nonzero
        # supercell translation, because an atom's own periodic image enters
        # the neighbor list there.  Minkowski reduction makes this bound
        # reliable for strongly skew cells as well as orthogonal cells.
        upper = float(np.min(periodic_lengths)) + 1e-8
        first, second, shifts, distances = neighbor_list(
            "ijSd", reference, upper, self_interaction=False
        )
        by_pair: dict[tuple[int, int], list[tuple[float, tuple[int, int, int]]]] = {}
        for atom_i, atom_j, shift, distance in zip(first, second, shifts, distances, strict=True):
            by_pair.setdefault((int(atom_i), int(atom_j)), []).append(
                (float(distance), tuple(int(value) for value in shift))
            )
        boundaries = []
        for (atom_i, atom_j), images in by_pair.items():
            ordered = sorted(set(images))
            if atom_i == atom_j:
                if ordered:
                    boundaries.append(ordered[0][0])
            elif len(ordered) > 1:
                boundaries.append(ordered[1][0])
        if not boundaries:
            raise RuntimeError("could not determine a finite-cell cutoff boundary")
        resolved = min(boundaries) - 0.01
        if resolved <= 0:
            raise ValueError("reference supercell is too small for cutoff=None")
        return float(resolved)
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
                shells.append(unique_periodic_distances(distances[first == site]))
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
    cutoff: float | None,
    max_body_order: int | None,
    symprec: float,
    tolerance: float = 1e-9,
    symmetry: PrimitiveSymmetryOperations | None = None,
) -> PrimitiveInteractionSpace:
    """Enumerate and symmetry-reduce exact primitive-lattice interactions."""
    if order < 2:
        raise ValueError("order must be at least two")
    if max_body_order is not None and not 1 <= max_body_order <= order:
        raise ValueError("max_body_order must be between 1 and order")
    primitive = primitive.copy()
    primitive.wrap()
    radius = resolve_primitive_cutoff(primitive, cutoff)
    if symmetry is None:
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
            basis = np.linalg.solve(basis[pivots].T, basis.T).T
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


def _canonical_key(key: InteractionKey, symmetry: PrimitiveSymmetryOperations) -> InteractionKey:
    best: InteractionKey | None = None
    for operation in range(symmetry.size):
        transformed = tuple(symmetry.transform_label(operation, label) for label in key.labels)
        for anchor in range(key.order):
            ordered = (transformed[anchor],) + transformed[:anchor] + transformed[anchor + 1 :]
            candidate = InteractionKey.from_labels((ordered[0], *sorted(ordered[1:])))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best
