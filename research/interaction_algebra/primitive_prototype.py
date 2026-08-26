"""Generator-built primitive interaction spaces and exhaustive comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from time import perf_counter

import numpy as np
from ase import Atoms
from indexed_orbit import IndexedOrbitResult, traverse_indexed_orbit
from scipy.linalg import qr
from sympy.combinatorics import Permutation, PermutationGroup

from mlfcs.interactions.algebra.actions import TensorAction
from mlfcs.interactions.algebra.invariants import (
    _label_symmetric_basis,
    invariant_basis_from_gram,
    normalize_pivot_basis,
    select_independent_rows,
)
from mlfcs.interactions.keys import InteractionKey
from mlfcs.interactions.models import (
    PrimitiveInteractionOrbit,
    PrimitiveInteractionSpace,
    PrimitiveOrbitImage,
)
from mlfcs.interactions.primitive.candidates import (
    _compatible_tails,
    _primitive_neighbors,
    resolve_primitive_cutoff,
)
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations


def encode_key(key: InteractionKey) -> np.ndarray:
    return np.asarray(key.labels, dtype=np.int64).reshape(-1)


def decode_key(values: np.ndarray) -> InteractionKey:
    rows = np.asarray(values, dtype=np.int64).reshape(-1, 4)
    return InteractionKey.from_labels(rows)


def _operation_signature(symmetry: PrimitiveSymmetryOperations, operation: int) -> tuple:
    shifts = np.asarray(symmetry.site_shifts[operation], dtype=np.int64)
    shifts = shifts - shifts[0]
    return (
        tuple(np.asarray(symmetry.rotations[operation], dtype=np.int64).reshape(-1)),
        tuple(np.asarray(symmetry.site_permutations[operation], dtype=np.int64)),
        tuple(shifts.reshape(-1)),
    )


def operation_composition_table(symmetry: PrimitiveSymmetryOperations) -> np.ndarray:
    """Build the exact affine operation table used to create a SymPy group."""
    lookup = {_operation_signature(symmetry, operation): operation for operation in range(symmetry.size)}
    table = np.empty((symmetry.size, symmetry.size), dtype=np.int32)
    for after in range(symmetry.size):
        for before in range(symmetry.size):
            before_sites = symmetry.site_permutations[before]
            rotation = symmetry.rotations[after] @ symmetry.rotations[before]
            sites = symmetry.site_permutations[after, before_sites]
            shifts = (
                symmetry.site_shifts[before] @ symmetry.rotations[after].T
                + symmetry.site_shifts[after, before_sites]
            )
            shifts -= shifts[0]
            signature = (
                tuple(np.asarray(rotation, dtype=np.int64).reshape(-1)),
                tuple(np.asarray(sites, dtype=np.int64)),
                tuple(np.asarray(shifts, dtype=np.int64).reshape(-1)),
            )
            table[after, before] = lookup[signature]
    return table


def sympy_space_group_generators(
    symmetry: PrimitiveSymmetryOperations,
) -> tuple[tuple[int, ...], PermutationGroup]:
    """Select deterministic affine generators using SymPy group orders."""
    table = operation_composition_table(symmetry)
    identity = int(np.flatnonzero(np.all(table == np.arange(symmetry.size), axis=1))[0])
    regular = tuple(Permutation([int(table[operation, value]) for value in range(symmetry.size)]) for operation in range(symmetry.size))
    selected: list[int] = []
    group = PermutationGroup([regular[identity]])
    while group.order() < symmetry.size:
        candidates = []
        for operation in range(symmetry.size):
            if operation in selected or operation == identity:
                continue
            candidate = PermutationGroup([*(regular[value] for value in selected), regular[operation]])
            candidates.append((int(candidate.order()), -operation, operation, candidate))
        _size, _negative, operation, group = max(candidates, key=lambda value: value[:2])
        selected.append(operation)
    group.schreier_sims()
    if group.order() != symmetry.size:
        raise RuntimeError("SymPy regular permutation group has the wrong order")
    return tuple(selected), group


@dataclass(frozen=True, slots=True)
class PrimitiveGenerator:
    name: str
    symmetry: PrimitiveSymmetryOperations
    operation: int | None
    permutation: tuple[int, ...]
    action: TensorAction

    def transform(self, states: np.ndarray) -> np.ndarray:
        result = np.empty_like(states)
        for row, values in enumerate(states):
            labels = np.asarray(values, dtype=np.int64).reshape(-1, 4)
            if self.operation is not None:
                transformed = np.empty_like(labels)
                sites = labels[:, 0].astype(np.int32)
                transformed[:, 0] = self.symmetry.site_permutations[self.operation, sites]
                transformed[:, 1:] = (
                    labels[:, 1:] @ self.symmetry.rotations[self.operation].T
                    + self.symmetry.site_shifts[self.operation, sites]
                )
                labels = transformed
            labels = labels[np.asarray(self.permutation, dtype=np.int32)]
            labels[:, 1:] -= labels[0, 1:]
            result[row] = labels.reshape(-1)
        return result


def primitive_generators(
    symmetry: PrimitiveSymmetryOperations, order: int
) -> tuple[tuple[PrimitiveGenerator, ...], PermutationGroup]:
    operations, group = sympy_space_group_generators(symmetry)
    identity = tuple(range(order))
    generators = [
        PrimitiveGenerator(
            f"space[{operation}]",
            symmetry,
            operation,
            identity,
            TensorAction(symmetry.cartesian_rotations[operation].T, identity, order),
        )
        for operation in operations
    ]
    for axis in range(order - 1):
        permutation = list(identity)
        permutation[axis], permutation[axis + 1] = permutation[axis + 1], permutation[axis]
        value = tuple(permutation)
        generators.append(
            PrimitiveGenerator(
                f"swap[{axis},{axis + 1}]",
                symmetry,
                None,
                value,
                TensorAction(np.eye(3), value, order),
            )
        )
    return tuple(generators), group


@dataclass(frozen=True, slots=True)
class GeneratedPrimitiveOrbit:
    representative: InteractionKey
    basis: np.ndarray
    pivots: np.ndarray
    result: IndexedOrbitResult


def generated_orbit(
    seed: InteractionKey,
    generators: tuple[PrimitiveGenerator, ...],
    *,
    tolerance: float,
) -> GeneratedPrimitiveOrbit:
    label_basis = _label_symmetric_basis(seed.labels)
    result = traverse_indexed_orbit(
        encode_key(seed),
        generators,
        order=seed.order,
        seed_basis=label_basis,
        canonical_columns=tuple(range(0, seed.order * 4, 4))
        + tuple(
            column
            for axis in range(1, seed.order)
            for column in range(axis * 4 + 1, axis * 4 + 4)
        )
        + tuple(range(1, 4)),
        tolerance=tolerance,
    )
    reduced = (
        invariant_basis_from_gram(result.constraint_gram, tolerance=tolerance)
        if np.any(result.constraint_gram)
        else np.eye(label_basis.shape[1])
    )
    seed_invariant = label_basis @ reduced
    canonical_invariant = result.seed_to_canonical.apply_columns(seed_invariant)
    pivots = select_independent_rows(canonical_invariant, tolerance=tolerance)
    basis = normalize_pivot_basis(canonical_invariant, pivots)
    return GeneratedPrimitiveOrbit(decode_key(result.canonical), basis, pivots, result)


def _projector_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[1] != right.shape[1]:
        return float("inf")
    q_left = qr(left, mode="economic", check_finite=False)[0]
    q_right = qr(right, mode="economic", check_finite=False)[0]
    return float(np.linalg.norm(q_left @ q_left.T - q_right @ q_right.T, ord=2))


def build_exhaustive_reference(
    primitive: Atoms,
    *,
    order: int,
    cutoff: float,
    max_body_order: int | None,
    symmetry: PrimitiveSymmetryOperations,
    tolerance: float,
) -> PrimitiveInteractionSpace:
    """Build the independent exhaustive oracle used only by this research."""
    radius = resolve_primitive_cutoff(primitive, cutoff)
    neighbors = _primitive_neighbors(primitive, radius)
    axis_permutations = tuple(permutations(range(order)))
    seen: set[InteractionKey] = set()
    orbits = []
    for anchor in range(len(primitive)):
        for tail in _compatible_tails(neighbors[anchor], order - 1, primitive, radius):
            key = InteractionKey.from_labels(((anchor, 0, 0, 0), *tail))
            if max_body_order is not None and len(set(key.labels)) > max_body_order:
                continue
            images: dict[InteractionKey, TensorAction] = {}
            for operation in range(symmetry.size):
                transformed = tuple(
                    symmetry.transform_label(operation, label) for label in key.labels
                )
                for permutation in axis_permutations:
                    image = InteractionKey.from_labels(
                        transformed[axis] for axis in permutation
                    )
                    images.setdefault(
                        image,
                        TensorAction(
                            symmetry.cartesian_rotations[operation].T,
                            permutation,
                            order,
                        ),
                    )
            representative = min(images)
            if representative in seen:
                continue
            seen.add(representative)
            label_basis = _label_symmetric_basis(representative.labels)
            gram = np.zeros((label_basis.shape[1],) * 2)
            representative_images: dict[InteractionKey, TensorAction] = {}
            for operation in range(symmetry.size):
                transformed = tuple(
                    symmetry.transform_label(operation, label)
                    for label in representative.labels
                )
                for permutation in axis_permutations:
                    image = InteractionKey.from_labels(
                        transformed[axis] for axis in permutation
                    )
                    action = TensorAction(
                        symmetry.cartesian_rotations[operation].T,
                        permutation,
                        order,
                    )
                    representative_images.setdefault(image, action)
                    if image == representative:
                        residual = action.apply_columns(label_basis) - label_basis
                        gram += residual.T @ residual
            basis = label_basis @ invariant_basis_from_gram(gram, tolerance=tolerance)
            if basis.shape[1] == 0:
                continue
            pivots = select_independent_rows(basis, tolerance=tolerance)
            basis = normalize_pivot_basis(basis, pivots)
            orbit_images = tuple(
                PrimitiveOrbitImage(image, action)
                for image, action in sorted(representative_images.items())
            )
            orbits.append(
                PrimitiveInteractionOrbit(representative, basis, pivots, orbit_images)
            )
    return PrimitiveInteractionSpace(
        primitive.copy(), order, radius, max_body_order, symmetry, tuple(orbits)
    )


def validate_primitive_case(
    primitive: Atoms,
    *,
    order: int,
    cutoff: float,
    max_body_order: int | None,
    symprec: float,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    symmetry = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=symprec)
    generators, sympy_group = primitive_generators(symmetry, order)
    started = perf_counter()
    expected = build_exhaustive_reference(
        primitive,
        order=order,
        cutoff=cutoff,
        max_body_order=max_body_order,
        tolerance=tolerance,
        symmetry=symmetry,
    )
    exhaustive_seconds = perf_counter() - started

    radius = resolve_primitive_cutoff(primitive, cutoff)
    neighbors = _primitive_neighbors(primitive, radius)
    generated: dict[InteractionKey, GeneratedPrimitiveOrbit] = {}
    known_representative: dict[InteractionKey, InteractionKey] = {}
    candidate_count = 0
    started = perf_counter()
    for anchor in range(len(primitive)):
        for tail in _compatible_tails(neighbors[anchor], order - 1, primitive, radius):
            seed = InteractionKey.from_labels(((anchor, 0, 0, 0), *tail))
            if max_body_order is not None and len(set(seed.labels)) > max_body_order:
                continue
            candidate_count += 1
            if seed in known_representative:
                continue
            orbit = generated_orbit(seed, generators, tolerance=tolerance)
            generated.setdefault(orbit.representative, orbit)
            for row in orbit.result.states:
                known_representative[decode_key(row)] = orbit.representative
    generator_seconds = perf_counter() - started

    expected_by_key = {orbit.representative: orbit for orbit in expected.orbits}
    if set(generated) != set(expected_by_key):
        missing = sorted(set(expected_by_key) - set(generated))
        extra = sorted(set(generated) - set(expected_by_key))
        raise AssertionError(
            "generated representative set differs from exhaustive production: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    maximum_projector_error = 0.0
    maximum_basis_error = 0.0
    action_errors = []
    edges = 0
    constraints = 0
    stabilizer_actions = 0
    for key, actual in generated.items():
        reference = expected_by_key[key]
        actual_keys = tuple(decode_key(row) for row in actual.result.states)
        expected_images = {image.key: image.action for image in reference.images}
        if set(actual_keys) != set(expected_images):
            raise AssertionError(f"image keys differ for {key}")
        if not np.array_equal(actual.pivots, reference.pivots):
            raise AssertionError(f"pivot indices differ for {key}")
        maximum_projector_error = max(
            maximum_projector_error, _projector_error(actual.basis, reference.basis)
        )
        maximum_basis_error = max(
            maximum_basis_error, float(np.max(np.abs(actual.basis - reference.basis), initial=0.0))
        )
        for image_key, action in zip(actual_keys, actual.result.actions, strict=True):
            # Transports to one image are defined only modulo the representative
            # stabilizer. They must agree on the invariant parameter space, not
            # on an arbitrary non-invariant Cartesian tensor.
            generated_columns = action.apply_columns(actual.basis)
            expected_columns = expected_images[image_key].apply_columns(actual.basis)
            error = np.max(np.abs(generated_columns - expected_columns), initial=0.0)
            action_errors.append(float(error))
        edges += actual.result.traversed_edges
        constraints += actual.result.schreier_constraints
        stabilizer_actions += actual.result.unique_stabilizer_actions
    maximum_action_error = max(action_errors, default=0.0)
    if maximum_action_error > 1e-9:
        raise AssertionError(f"canonical image action error is {maximum_action_error:.3e}")
    return {
        "order": order,
        "candidate_count": candidate_count,
        "orbit_count": len(generated),
        "parameter_count": sum(orbit.basis.shape[1] for orbit in generated.values()),
        "image_count": sum(len(orbit.result.states) for orbit in generated.values()),
        "space_group_order": symmetry.size,
        "sympy_group_order": int(sympy_group.order()),
        "space_generator_count": len(generators) - order + 1,
        "total_generator_count": len(generators),
        "traversed_edges": edges,
        "schreier_constraints": constraints,
        "unique_stabilizer_actions": stabilizer_actions,
        "maximum_projector_error": maximum_projector_error,
        "maximum_normalized_basis_error": maximum_basis_error,
        "maximum_action_error": maximum_action_error,
        "exhaustive_seconds": exhaustive_seconds,
        "indexed_generator_seconds": generator_seconds,
    }


__all__ = [
    "GeneratedPrimitiveOrbit",
    "decode_key",
    "encode_key",
    "generated_orbit",
    "primitive_generators",
    "sympy_space_group_generators",
    "validate_primitive_case",
]
