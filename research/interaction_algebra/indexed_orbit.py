"""NumPy-indexed generator orbit traversal for research prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mlfcs.interactions.algebra.actions import TensorAction


class IndexedGenerator(Protocol):
    """One exact state action and its Cartesian tensor representation."""

    name: str
    action: TensorAction

    def transform(self, states: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class IndexedOrbitResult:
    """One canonical orbit with transports and Schreier constraint Gram."""

    canonical: np.ndarray
    states: np.ndarray
    actions: tuple[TensorAction, ...]
    seed_to_canonical: TensorAction
    constraint_gram: np.ndarray
    traversed_edges: int
    schreier_constraints: int
    unique_stabilizer_actions: int


def compose_actions(after: TensorAction, before: TensorAction) -> TensorAction:
    """Return the tensor action ``after(before(values))``."""
    if after.order != before.order:
        raise ValueError("cannot compose tensor actions of different orders")
    permutation = tuple(before.permutation[axis] for axis in after.permutation)
    return TensorAction(after.rotation @ before.rotation, permutation, after.order)


def inverse_action(action: TensorAction) -> TensorAction:
    """Return the inverse of an orthogonal Cartesian/permutation action."""
    inverse_permutation = tuple(int(value) for value in np.argsort(action.permutation))
    return TensorAction(np.linalg.inv(action.rotation), inverse_permutation, action.order)


def _row_keys(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


def _lookup_rows(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    """Return row positions or -1 using NumPy sorting rather than Python hashes."""
    if not len(needles):
        return np.empty(0, dtype=np.int64)
    if not len(haystack):
        return np.full(len(needles), -1, dtype=np.int64)
    haystack_keys = _row_keys(haystack)
    needle_keys = _row_keys(needles)
    order = np.argsort(haystack_keys)
    sorted_keys = haystack_keys[order]
    positions = np.searchsorted(sorted_keys, needle_keys)
    found = positions < len(sorted_keys)
    safe = np.minimum(positions, len(sorted_keys) - 1)
    found &= sorted_keys[safe] == needle_keys
    result = np.full(len(needles), -1, dtype=np.int64)
    result[found] = order[safe[found]]
    return result


def _canonical_order(states: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    return np.lexsort(tuple(states[:, axis] for axis in reversed(columns)))


def _action_signature(action: TensorAction) -> tuple[tuple[float, ...], tuple[int, ...]]:
    rotation = tuple(float(value) for value in np.round(action.rotation, decimals=12).reshape(-1))
    return rotation, action.permutation


def traverse_indexed_orbit(
    seed: np.ndarray,
    generators: tuple[IndexedGenerator, ...],
    *,
    order: int,
    seed_basis: np.ndarray | None = None,
    canonical_columns: tuple[int, ...] | None = None,
    tolerance: float = 1e-9,
) -> IndexedOrbitResult:
    """Traverse an orbit with integer rows and re-anchor transports canonically.

    Dynamic states are discovered in NumPy batches. Membership and duplicate
    detection use sorted fixed-width rows; no ``InteractionKey`` hash table is
    maintained in the traversal loop.
    """
    seed = np.asarray(seed, dtype=np.int64).reshape(1, -1)
    identity = TensorAction(np.eye(3), tuple(range(order)), order)
    basis = np.eye(3**order) if seed_basis is None else np.asarray(seed_basis, dtype=float)
    states = seed.copy()
    transports: list[TensorAction] = [identity]
    frontier = np.asarray([0], dtype=np.int64)
    traversed_edges = 0
    schreier_constraints = 0
    stabilizers: dict[tuple[tuple[float, ...], tuple[int, ...]], TensorAction] = {}

    while len(frontier):
        source_states = states[frontier]
        candidate_rows = []
        candidate_actions: list[TensorAction] = []
        for generator in generators:
            transformed = np.asarray(generator.transform(source_states), dtype=np.int64)
            if transformed.shape != source_states.shape:
                raise ValueError(f"generator {generator.name} returned shape {transformed.shape}")
            candidate_rows.append(transformed)
            candidate_actions.extend(
                compose_actions(generator.action, transports[int(source)]) for source in frontier
            )
        candidates = np.vstack(candidate_rows)
        traversed_edges += len(candidates)
        known = _lookup_rows(states, candidates)

        unknown_locations = np.flatnonzero(known < 0)
        new_indices: list[int] = []
        if len(unknown_locations):
            unknown_rows = candidates[unknown_locations]
            unique_rows, first, inverse = np.unique(
                unknown_rows, axis=0, return_index=True, return_inverse=True
            )
            first_locations = unknown_locations[first]
            base = len(states)
            states = np.vstack((states, unique_rows))
            transports.extend(candidate_actions[int(location)] for location in first_locations)
            new_indices.extend(range(base, len(states)))
            known[unknown_locations] = base + inverse

        for location, target in enumerate(known):
            candidate_action = candidate_actions[location]
            previous_action = transports[int(target)]
            if (
                candidate_action.permutation == previous_action.permutation
                and np.allclose(candidate_action.rotation, previous_action.rotation, atol=tolerance)
            ):
                continue
            stabilizer = compose_actions(inverse_action(previous_action), candidate_action)
            stabilizers.setdefault(_action_signature(stabilizer), stabilizer)
            schreier_constraints += 1
        frontier = np.asarray(new_indices, dtype=np.int64)

    gram = np.zeros((basis.shape[1], basis.shape[1]), dtype=float)
    for stabilizer in stabilizers.values():
        residual = stabilizer.apply_columns(basis) - basis
        if np.linalg.norm(residual) > tolerance:
            gram += residual.T @ residual

    columns = tuple(range(states.shape[1])) if canonical_columns is None else canonical_columns
    if sorted(columns) != list(range(states.shape[1])):
        raise ValueError("canonical_columns must be a permutation of state columns")
    canonical_order = _canonical_order(states, columns)
    canonical_index = int(canonical_order[0])
    canonical_inverse = inverse_action(transports[canonical_index])
    anchored = tuple(compose_actions(action, canonical_inverse) for action in transports)
    sorted_states = states[canonical_order]
    sorted_actions = tuple(anchored[int(index)] for index in canonical_order)
    return IndexedOrbitResult(
        canonical=sorted_states[0].copy(),
        states=sorted_states,
        actions=sorted_actions,
        seed_to_canonical=transports[canonical_index],
        constraint_gram=gram,
        traversed_edges=traversed_edges,
        schreier_constraints=schreier_constraints,
        unique_stabilizer_actions=len(stabilizers),
    )


__all__ = [
    "IndexedGenerator",
    "IndexedOrbitResult",
    "compose_actions",
    "inverse_action",
    "traverse_indexed_orbit",
]
