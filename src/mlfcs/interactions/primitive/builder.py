"""Generator-built primitive interaction spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from sympy.combinatorics import Permutation, PermutationGroup

from mlfcs.interactions.algebra.actions import TensorAction
from mlfcs.interactions.algebra.generators import select_group_generators
from mlfcs.interactions.algebra.indexed_orbit import (
    IndexedOrbitResult,
    traverse_indexed_orbit,
)
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
from mlfcs.interactions.primitive.candidates import resolve_primitive_cutoff
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations


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
    """Build primitive orbits through indexed generator traversal."""
    from mlfcs.interactions.primitive.candidates import iter_primitive_candidates

    primitive = primitive.copy()
    primitive.wrap()
    radius = resolve_primitive_cutoff(primitive, cutoff)
    symmetry = symmetry or PrimitiveSymmetryOperations.from_atoms(primitive, symprec=symprec)
    generators, _group = primitive_generators(symmetry, order)
    generated = {}
    covered = {}
    for seed in iter_primitive_candidates(
        primitive, radius=radius, order=order, max_body_order=max_body_order
    ):
        if seed in covered:
            continue
        orbit = generated_orbit(seed, generators, tolerance=tolerance)
        generated[orbit.representative] = orbit
        for row in orbit.result.states:
            covered[decode_key(row)] = orbit.representative
    result = []
    for representative in sorted(generated):
        orbit = generated[representative]
        if orbit.basis.shape[1] == 0:
            continue
        images = tuple(
            PrimitiveOrbitImage(decode_key(row), action)
            for row, action in zip(orbit.result.states, orbit.result.actions, strict=True)
        )
        result.append(
            PrimitiveInteractionOrbit(
                orbit.representative, orbit.basis, orbit.pivots, images
            )
        )
    return PrimitiveInteractionSpace(
        primitive, order, radius, max_body_order, symmetry, tuple(result)
    )


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
    regular = tuple(Permutation([int(table[operation, value]) for value in range(symmetry.size)]) for operation in range(symmetry.size))
    selected_permutations = select_group_generators(regular)
    selected = tuple(regular.index(permutation) for permutation in selected_permutations)
    group = PermutationGroup(list(selected_permutations))
    group.schreier_sims()
    if group.order() != symmetry.size:
        raise RuntimeError("SymPy regular permutation group has the wrong order")
    return selected, group


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


__all__ = [
    "GeneratedPrimitiveOrbit",
    "decode_key",
    "encode_key",
    "generated_orbit",
    "primitive_generators",
    "sympy_space_group_generators",
]
