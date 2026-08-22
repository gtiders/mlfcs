#!/usr/bin/env python3
"""Validate generator-based primitive interaction orbits against production enumeration."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from math import factorial
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read
from scipy.linalg import eigh, qr

from mlfcs.interactions.tensors import TensorAction, _label_symmetric_basis
from mlfcs.interactions.enumerate import build_primitive_interaction_space
from mlfcs.interactions.keys import InteractionKey
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class ActionGenerator:
    name: str
    operation: int | None
    permutation: tuple[int, ...]
    action: TensorAction


def _operation_signature(symmetry: PrimitiveSymmetryOperations, operation: int) -> tuple:
    shifts = np.asarray(symmetry.site_shifts[operation], dtype=np.int64)
    relative_shifts = shifts - shifts[0]
    return (
        tuple(np.asarray(symmetry.rotations[operation], dtype=np.int64).reshape(-1)),
        tuple(np.asarray(symmetry.site_permutations[operation], dtype=np.int64)),
        tuple(relative_shifts.reshape(-1)),
    )


def _composition_table(symmetry: PrimitiveSymmetryOperations) -> np.ndarray:
    """Return ``table[after, before]`` for exact affine label actions."""
    lookup = {
        _operation_signature(symmetry, operation): operation
        for operation in range(symmetry.size)
    }
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
            shifts = shifts - shifts[0]
            signature = (
                tuple(np.asarray(rotation, dtype=np.int64).reshape(-1)),
                tuple(np.asarray(sites, dtype=np.int64)),
                tuple(np.asarray(shifts, dtype=np.int64).reshape(-1)),
            )
            try:
                table[after, before] = lookup[signature]
            except KeyError as error:
                raise RuntimeError("spglib operations are not closed as exact affine actions") from error
    return table


def _identity_operation(symmetry: PrimitiveSymmetryOperations) -> int:
    identity_sites = np.arange(len(symmetry.site_permutations[0]), dtype=np.int32)
    for operation in range(symmetry.size):
        if (
            np.array_equal(symmetry.rotations[operation], np.eye(3, dtype=np.int32))
            and np.array_equal(symmetry.site_permutations[operation], identity_sites)
            and not np.any(symmetry.site_shifts[operation])
        ):
            return operation
    raise RuntimeError("space group has no explicit identity operation")


def _generated_subgroup(
    table: np.ndarray, identity: int, generators: tuple[int, ...]
) -> frozenset[int]:
    reached = {identity}
    queue = deque((identity,))
    while queue:
        current = queue.popleft()
        for generator in generators:
            image = int(table[generator, current])
            if image not in reached:
                reached.add(image)
                queue.append(image)
    return frozenset(reached)


def minimal_greedy_generators(symmetry: PrimitiveSymmetryOperations) -> tuple[int, ...]:
    """Choose a deterministic small generating set from the complete operation list."""
    table = _composition_table(symmetry)
    identity = _identity_operation(symmetry)
    selected: tuple[int, ...] = ()
    reached = _generated_subgroup(table, identity, selected)
    while len(reached) != symmetry.size:
        candidates = []
        for operation in range(symmetry.size):
            if operation in reached:
                continue
            closure = _generated_subgroup(table, identity, (*selected, operation))
            candidates.append((len(closure), -operation, operation, closure))
        _size, _negative, operation, closure = max(candidates)
        selected = (*selected, operation)
        reached = closure
    return selected


def _transform_key(
    key: InteractionKey,
    symmetry: PrimitiveSymmetryOperations,
    generator: ActionGenerator,
) -> InteractionKey:
    labels = key.labels
    if generator.operation is not None:
        labels = tuple(
            symmetry.transform_label(generator.operation, label) for label in labels
        )
    return InteractionKey.from_labels(labels[axis] for axis in generator.permutation)


def _action_generators(
    symmetry: PrimitiveSymmetryOperations, order: int
) -> tuple[ActionGenerator, ...]:
    identity = tuple(range(order))
    generators = [
        ActionGenerator(
            f"space[{operation}]",
            operation,
            identity,
            TensorAction(symmetry.cartesian_rotations[operation].T, identity, order),
        )
        for operation in minimal_greedy_generators(symmetry)
    ]
    for axis in range(order - 1):
        permutation = list(identity)
        permutation[axis], permutation[axis + 1] = permutation[axis + 1], permutation[axis]
        value = tuple(permutation)
        generators.append(
            ActionGenerator(
                f"swap[{axis},{axis + 1}]",
                None,
                value,
                TensorAction(np.eye(3), value, order),
            )
        )
    return tuple(generators)


def generator_orbit(
    representative: InteractionKey,
    symmetry: PrimitiveSymmetryOperations,
    generators: tuple[ActionGenerator, ...],
    *,
    tolerance: float = 1e-9,
) -> tuple[set[InteractionKey], np.ndarray, int]:
    """Traverse one key orbit and derive invariant tensors from Schreier edges."""
    label_basis = _label_symmetric_basis(representative.labels)
    states = {representative: label_basis}
    queue = deque((representative,))
    gram = np.zeros((label_basis.shape[1], label_basis.shape[1]))
    nontrivial_edges = 0
    while queue:
        key = queue.popleft()
        columns = states[key]
        for generator in generators:
            image = _transform_key(key, symmetry, generator)
            candidate = generator.action.apply_columns(columns)
            known = states.get(image)
            if known is None:
                states[image] = candidate
                queue.append(image)
                continue
            constraint = candidate - known
            if np.max(np.abs(constraint), initial=0.0) > tolerance:
                gram += constraint.T @ constraint
                nontrivial_edges += 1
    values, vectors = eigh(0.5 * (gram + gram.T), check_finite=False, driver="evr")
    threshold = tolerance * max(float(np.max(np.abs(values), initial=0.0)), 1.0)
    invariant = label_basis @ vectors[:, values <= threshold]
    return set(states), invariant, nontrivial_edges


def _subspace_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[1] != right.shape[1]:
        return float("inf")
    if left.shape[1] == 0:
        return 0.0
    q_left = qr(left, mode="economic", pivoting=False, check_finite=False)[0]
    q_right = qr(right, mode="economic", pivoting=False, check_finite=False)[0]
    return float(np.linalg.norm(q_left @ q_left.T - q_right @ q_right.T, ord=2))


def validate_case(
    name: str,
    primitive: Atoms,
    specifications: tuple[tuple[int, float, int | None], ...],
    *,
    symprec: float,
) -> None:
    symmetry = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=symprec)
    space_generators = minimal_greedy_generators(symmetry)
    print(
        f"{name}: space_group={symmetry.symbol}, operations={symmetry.size}, "
        f"space_generators={space_generators}"
    )
    for order, cutoff, body in specifications:
        space = build_primitive_interaction_space(
            primitive,
            order=order,
            cutoff=cutoff,
            max_body_order=body,
            symprec=symprec,
            symmetry=symmetry,
        )
        generators = _action_generators(symmetry, order)
        maximum_subspace_error = 0.0
        image_count = 0
        schreier_edges = 0
        for orbit in space.orbits:
            keys, invariant, edges = generator_orbit(
                orbit.representative, symmetry, generators
            )
            expected = {image.key for image in orbit.images}
            if keys != expected:
                raise AssertionError(
                    f"{name} FC{order} orbit {orbit.representative}: image-key mismatch"
                )
            if invariant.shape[1] != orbit.dimension:
                raise AssertionError(
                    f"{name} FC{order} orbit {orbit.representative}: "
                    f"generator dimension {invariant.shape[1]} != {orbit.dimension}"
                )
            maximum_subspace_error = max(
                maximum_subspace_error, _subspace_error(invariant, orbit.basis)
            )
            image_count += len(keys)
            schreier_edges += edges
        exhaustive_actions = len(space.orbits) * symmetry.size * factorial(order)
        generator_edges = image_count * len(generators)
        print(
            f"  FC{order}: orbits={len(space.orbits)}, parameters="
            f"{sum(orbit.dimension for orbit in space.orbits)}, images={image_count}, "
            f"generators={len(generators)}, exhaustive_actions={exhaustive_actions}, "
            f"generator_edges={generator_edges}, schreier_constraints={schreier_edges}, "
            f"maximum_subspace_error={maximum_subspace_error:.3e}"
        )


def _cases() -> dict[str, tuple[Atoms, tuple[tuple[int, float, int | None], ...], float]]:
    return {
        "si": (
            bulk("Si", "diamond", a=5.43),
            ((2, 5.4, 2), (3, 5.4, 3), (4, 4.6, 3)),
            1e-5,
        ),
        "snse": (
            read(ROOT / "examples/fitting/SnSe/input/primitive.vasp"),
            ((2, 8.0, 2), (3, 6.5, 3), (4, 4.5, 3)),
            1e-4,
        ),
        "ba8ga16ge30": (
            read(ROOT / "examples/fitting/Ba8Ga16Ge30/input/primitive.vasp"),
            ((2, 5.4, 2), (3, 4.35, 2), (4, 4.35, 2)),
            1e-4,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", choices=tuple(_cases()), default=tuple(_cases()))
    arguments = parser.parse_args()
    cases = _cases()
    for name in arguments.cases:
        primitive, specifications, symprec = cases[name]
        validate_case(name, primitive, specifications, symprec=symprec)


if __name__ == "__main__":
    main()
